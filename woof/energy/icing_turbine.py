"""Wind-turbine icing at hub height: the ``turbine-icing`` product.

The ``icing`` product in :mod:`woof.energy.products` accretes ice on line
conductors, and for every other site on the ISO 12494 30 mm reference
collector at 10 m.  A wind turbine ices at its hub and on its rotating
blades, so this module diagnoses icing at each turbine's ``hub_height_m``
(site series, ``woof energy rating --products turbine-icing``) and on any
grid of model output (:func:`turbine_icing_dataset`, xarray in and out, for
wrfout files and the Icechunk hindcast stores).  The two paths share one
array kernel, :func:`turbine_icing_arrays`, so a grid point and a site with
the same weather get the same numbers.

What is computed (every constant below is a module constant, written into
the output attributes; ``docs/energy-hindcast/icing.md`` is the guide):

Liquid water content
    ``w_c = rho q_c`` (cloud) and ``w_r = rho q_r`` (rain and drizzle, when
    the forecast carries ``rain_mixing_ratio`` and ``include_rain`` is set);
    the reported LWC is ``w_c + w_r``.

Droplet median volume diameter (MVD)
    The cloud droplet spectrum is Thompson's generalised gamma
    ``N(D) = N0 D^mu exp(-lambda D)`` with Thompson and Eidhammer (2014)'s
    shape ``mu = clip(nint(1e9 / N_c) + 2, 2, 15)`` (``N_c`` in m-3).  With
    ``LWC = (pi rho_w / 6) N_c (mu+3)(mu+2)(mu+1) / lambda^3`` the slope is
    ``lambda = [(pi rho_w / 6) (mu+1)(mu+2)(mu+3) N_c / w_c]^(1/3)`` and the
    MVD is the median of the mass spectrum (a gamma of shape ``mu + 4``),
    ``(mu + 3.67) / lambda``, clipped to [5, 50] um.  ``N_c = rho QNCLOUD``
    where the run carries Thompson's droplet number (``mp_physics = 28``,
    forecast variable ``cloud_number_mixing_ratio``); otherwise, and at a
    cloudy point whose droplet number is zero or missing, the documented
    default ``N_c = 1e8 m-3`` -- the fixed droplet number of Thompson
    ``mp_physics = 8`` -- is used and said so.  WSM6 carries no
    droplet number either.

Accretion (Makkonen 2000, ISO 12494)
    ``dM/dt = a1 a2 a3 w v D`` per metre of collector, ``a2 = 1``.  ``a1``
    for cloud droplets is Finstad, Lozowski and Gates (1988) at the MVD;
    rain drops (> 100 um) are collected with ``a1 = 1`` at the speed
    ``sqrt(v^2 + 4^2)`` m/s (an assumed 4 m/s fall speed, Jones 1998).
    ``a3`` is the frozen fraction of the Makkonen (2000) heat balance at a
    0 degC surface: convective cooling ``h pi D (0 - t_a)``, evaporative
    cooling ``h pi D eps L_v (e_s(0) - e_a) / (c_p p)`` (only with
    ``air_pressure``; ``e_a`` from ``relative_humidity`` or saturated air),
    the warming of the collected water ``F c_w (0 - t_a)``, against the
    aerodynamic heating ``h pi D r v^2 / (2 c_p)`` (recovery factor
    ``r = 0.79``), the droplets' kinetic energy ``F v^2 / 2`` and the
    latent heat of freezing ``F L_f``.  ``h = Nu k_a / D`` with Makkonen's
    ``Nu = 0.032 Re^0.85``.  Dry growth (``a3 = 1``) is stored as rime at
    500 kg m-3 and wet growth as glaze at 900 kg m-3; the collector diameter
    grows with the accreted volume.  Radiation and the latent heat of
    sublimation are neglected.

Removal
    When that heat balance has a surplus (the surface cannot stay at 0 degC
    -- warm air, or aerodynamic heating on a fast blade) the surplus melts
    ice at ``surplus / L_f``.  Below freezing, with ``air_pressure`` and
    ``relative_humidity`` present, ice sublimates at
    ``h pi D eps (e_si(t_a) - e_a) / (c_p p)`` (Chilton-Colburn, Lewis
    number 1, surface at air temperature), never negative, and only from a
    frozen surface (not during wet growth, whose evaporation is already in
    the heat balance).  There is no
    mechanical shedding; for rotating blades this overstates how long ice
    stays, and it is said in the attributes.

Two collectors at hub height
    * the ISO 12494 reference collector (30 mm cylinder) in the hub wind:
      its ice is the *instrumental icing* of IEA Wind Task 19 (ice present
      on a structure or instrument, at least 10 g/m);
    * a rotating-blade proxy (:class:`BladeProxy`): a cylinder of 0.10 m,
      standing in for the leading edge of the outer blade, at 85 % of the
      rotor radius.  While the turbine runs (cut-in to cut-out of the power
      curve) the tip speed is ``min(TSR v, 80 m/s)`` with tip-speed ratio
      7, and the section sees ``sqrt((0.85 v_tip)^2 + ((1 - a) v)^2)`` with
      the Betz axial induction ``a = 1/3``; parked (outside cut-in to
      cut-out) it sees the hub wind ``v``.  Its ice is *rotor icing*.

Meteorological icing
    ``t_a < 0 degC`` and ``LWC > 0.01 g m-3``: the conditions for active
    accretion (IEA Wind Task 19).

IEA Wind Task 19 ice class
    From the fraction of the period with meteorological icing:
    class 1 at most 0.5 %, 2 at most 3 %, 3 at most 5 %, 4 at most 10 %,
    5 above (IEA Wind Task 19 Recommended Practice 13, "Wind energy in
    cold climates").  The classes are defined on annual statistics; a
    shorter period is classed as if it were a year, and said so.

Production loss (a simple documented model, not a calibrated one)
    From the blade-proxy ice mass ``M`` (kg/m): no loss below the 10 g/m
    trace, ``0.5 M / 1.0 kg m-1`` while ``M`` is below the 1.0 kg/m stop
    load (aerodynamic degradation of the power curve, up to half), and a
    full stop (loss 1) at or above it (ice detection or rotor imbalance).
    The period's energy loss is ``sum(P L) / sum(P)``, ``P`` from the power
    curve at the density-corrected hub wind (IEC 61400-12).  Calibrate the
    three :class:`IcingLossModel` numbers against SCADA before relying on
    them; IEA Wind Task 19 reports AEP losses of up to 20 % and more in
    the heaviest ice classes.

Time stepping
    A rate at time ``t_n`` is diagnosed from the weather at ``t_n`` and the
    ice already on the collector, and held over the interval
    ``(t_{n-1}, t_n]`` -- the convention of ``precipitation_rate`` in
    ``woof energy extract``.  Mass at the first time is the initial state
    (zero, or ``initial_state`` for a chunk of a longer hindcast).  An
    unknown input (NaN) makes the ice unknown from then on, never zero.

Python boundary: like :mod:`woof.energy.products` this is element-wise
algebra on arrays the size of the output (a time loop over vectors), which
``docs/dev/static-rust-port.md`` allows in Python.  The kernel is written
against the array module of its inputs, so CuPy arrays stay on the GPU
(``device="gpu"`` in :func:`turbine_icing_dataset`); NumPy is the default.
Gridded inputs are mapped with :func:`xarray.apply_ufunc` over spatial dask
chunks (time must be one chunk; it is rechunked if not).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import numbers
import sys
from typing import Any, Mapping, Sequence

import numpy as np

from woof.energy.contracts import ContractError
from woof.energy.products import (ICE_TRACE_KG_M as ISO12494_TRACE_KG_M,
                                  ISO12494_RIME_LIMITS_KG_M)

TURBINE_ICING_SCHEMA = "woof-energy.turbine-icing.v1"
PRODUCT = "turbine-icing"

# physical constants
KELVIN = 273.15
RHO_WATER = 1000.0          # kg m-3
L_FUSION = 3.34e5           # J kg-1
L_EVAP = 2.501e6            # J kg-1
CP_AIR = 1004.0             # J kg-1 K-1
CW = 4218.0                 # J kg-1 K-1, supercooled water
EPSILON = 0.622             # R_d / R_v
RECOVERY_FACTOR = 0.79      # Makkonen (2000), cylinder
RIME_DENSITY = 500.0        # kg m-3
GLAZE_DENSITY = 900.0       # kg m-3

#: Droplet spectrum (Thompson; see the module docstring).
DEFAULT_CLOUD_NUMBER_M3 = 100e6
MVD_MIN_M = 5e-6
MVD_MAX_M = 50e-6
#: Rain and drizzle: collection efficiency 1 at this assumed fall speed.
RAIN_FALL_SPEED = 4.0       # m s-1

#: ISO 12494 reference collector.
REFERENCE_DIAMETER_M = 0.030
#: Meteorological icing: t < 0 degC and LWC above this (0.01 g m-3).
MET_ICING_LWC_KG_M3 = 1e-5
#: Ice present (instrumental and rotor icing): at least 10 g per metre.
ICE_PRESENT_KG_M = 0.01

#: IEA Wind Task 19 ice classes 1..5 by meteorological icing, upper bounds
#: in percent of the period (class 5 is anything above 10 %).
IEA_MET_ICING_LIMITS_PCT = (0.5, 3.0, 5.0, 10.0)


class TurbineIcingRefused(ContractError):
    """Turbine icing cannot be computed from these inputs."""


@dataclass(frozen=True)
class BladeProxy:
    """The rotating-blade collector (see the module docstring)."""

    tip_speed_ratio: float = 7.0
    max_tip_speed_m_s: float = 80.0
    cut_in_m_s: float = 3.0
    cut_out_m_s: float = 25.0
    radial_fraction: float = 0.85
    axial_induction: float = 1.0 / 3.0
    collector_diameter_m: float = 0.10

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not (isinstance(value, (int, float)) and math.isfinite(value)
                    and value >= 0.0):
                raise TurbineIcingRefused(
                    f"blade proxy {name} must be a finite number >= 0, got "
                    f"{value!r}")
        if not 0.0 < self.radial_fraction <= 1.0:
            raise TurbineIcingRefused("blade proxy radial_fraction must be "
                                      "in (0, 1]")
        if not self.axial_induction < 1.0:
            raise TurbineIcingRefused("blade proxy axial_induction must be "
                                      "below 1")
        if self.collector_diameter_m <= 0.0 or self.tip_speed_ratio <= 0.0:
            raise TurbineIcingRefused("blade proxy collector_diameter_m and "
                                      "tip_speed_ratio must be positive")
        if self.cut_out_m_s <= self.cut_in_m_s:
            raise TurbineIcingRefused("blade proxy cut_out_m_s must exceed "
                                      "cut_in_m_s")

    def describe(self) -> str:
        return (f"blade proxy: {self.collector_diameter_m:g} m cylinder at "
                f"{self.radial_fraction:g} R; running between "
                f"{self.cut_in_m_s:g} and {self.cut_out_m_s:g} m/s with tip "
                f"speed min({self.tip_speed_ratio:g} v, "
                f"{self.max_tip_speed_m_s:g} m/s) and axial induction "
                f"{self.axial_induction:.3g}; parked it sees the hub wind")


@dataclass(frozen=True)
class IcingLossModel:
    """Production loss from the blade-proxy ice mass (see the docstring)."""

    trace_kg_m: float = ICE_PRESENT_KG_M
    operating_loss_max: float = 0.5
    stop_kg_m: float = 1.0

    def __post_init__(self):
        if not (0.0 <= self.trace_kg_m < self.stop_kg_m
                and 0.0 <= self.operating_loss_max <= 1.0
                and math.isfinite(self.stop_kg_m)):
            raise TurbineIcingRefused(
                "loss model needs 0 <= trace_kg_m < stop_kg_m (finite) and "
                "operating_loss_max in [0, 1]")

    def describe(self) -> str:
        return (f"loss 0 below {self.trace_kg_m:g} kg/m of blade ice, "
                f"{self.operating_loss_max:g} M / {self.stop_kg_m:g} kg/m "
                f"below {self.stop_kg_m:g} kg/m, 1 (stopped) at or above")


DEFAULT_BLADE = BladeProxy()
DEFAULT_LOSS = IcingLossModel()


# --------------------------------------------------------------------------
# array module


def array_module(*arrays: Any):
    """``cupy`` when any input is a CuPy array (and CuPy is imported),
    else ``numpy``.  CuPy is never imported here."""

    cupy = sys.modules.get("cupy")
    if cupy is None:
        return np
    return cupy.get_array_module(*[a for a in arrays if a is not None])


def _to_numpy(array: Any) -> np.ndarray:
    if hasattr(array, "get") and not isinstance(array, np.ndarray):
        return array.get()
    return np.asarray(array)


# --------------------------------------------------------------------------
# microphysics


def thompson_cloud_shape(number_m3: Any) -> Any:
    """Thompson and Eidhammer (2014) droplet gamma shape ``mu`` (N in m-3)."""

    xp = array_module(number_m3)
    n = xp.asarray(number_m3, dtype=xp.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        # Fortran NINT of a positive number: half rounds up
        raw = xp.floor(1.0e9 / xp.where(n > 0, n, xp.nan) + 0.5) + 2.0
    return xp.clip(xp.where(xp.isfinite(raw), raw, 15.0), 2.0, 15.0)


def droplet_mvd(cloud_lwc: Any, number_m3: Any) -> Any:
    """Median volume diameter (m) of a Thompson gamma droplet spectrum.

    ``cloud_lwc`` in kg m-3, ``number_m3`` in m-3.  NaN where there is no
    cloud water or no droplet number; clipped to [MVD_MIN_M, MVD_MAX_M].
    """

    xp = array_module(cloud_lwc, number_m3)
    w = xp.asarray(cloud_lwc, dtype=xp.float64)
    n = xp.asarray(number_m3, dtype=xp.float64)
    mu = thompson_cloud_shape(n)
    ok = (w > 0) & (n > 0)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        lam = (math.pi * RHO_WATER / 6.0 * (mu + 1.0) * (mu + 2.0)
               * (mu + 3.0) * n / xp.where(ok, w, 1.0)) ** (1.0 / 3.0)
        mvd = (mu + 3.67) / lam
    return xp.where(ok, xp.clip(mvd, MVD_MIN_M, MVD_MAX_M), xp.nan)


def _viscosity(air_temp_k):
    return 1.458e-6 * air_temp_k ** 1.5 / (air_temp_k + 110.4)


def collision_efficiency(diameter_m: Any, speed: Any, air_temp_k: Any,
                         air_density: Any, mvd_m: Any) -> Any:
    """Finstad, Lozowski and Gates (1988) collision efficiency ``a1``.

    The same fit as :func:`woof.energy.products.finstad_collision_efficiency`
    with the MVD an array; 0 in calm air, NaN where speed or temperature is.
    """

    xp = array_module(diameter_m, speed, air_temp_k, air_density, mvd_m)
    mu = _viscosity(air_temp_k)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        k = RHO_WATER * mvd_m ** 2 * speed / (9.0 * mu * diameter_m)
        re = air_density * speed * mvd_m / mu
        langmuir = xp.where(k > 0, re ** 2 / k, 0.0)
        a = 1.066 * k ** -0.00616 * xp.exp(-1.103 * k ** -0.688)
        b = 3.641 * k ** -0.498 * xp.exp(-1.497 * k ** -0.694)
        c = 0.00637 * xp.maximum(langmuir - 100.0, 0.0) ** 0.381
        e = a - 0.028 - c * (b - 0.0454)
    e = xp.where((k > 0) & xp.isfinite(e), e, 0.0)
    e = xp.clip(e, 0.0, 1.0)
    return xp.where(xp.isfinite(speed) & xp.isfinite(air_temp_k), e, xp.nan)


def saturation_vapour_pressure_water(temp_c):
    """Magnus over liquid water, Pa (as :mod:`woof.energy.products`)."""

    xp = array_module(temp_c)
    return 611.2 * xp.exp(17.67 * temp_c / (temp_c + 243.5))


def saturation_vapour_pressure_ice(temp_c):
    """Magnus over ice, Pa: ``611.2 exp(22.46 t / (272.62 + t))``."""

    xp = array_module(temp_c)
    return 611.2 * xp.exp(22.46 * temp_c / (temp_c + 272.62))


def heat_transfer_coefficient(diameter_m, speed, air_temp_k, air_density):
    """Makkonen (2000) ``h = 0.032 Re^0.85 k_a / D`` (W m-2 K-1)."""

    xp = array_module(diameter_m, speed, air_temp_k, air_density)
    tc = air_temp_k - KELVIN
    k_air = 2.424e-2 + 7.477e-5 * tc - 4.407e-9 * tc ** 2
    re = air_density * xp.maximum(speed, 0.0) * diameter_m / _viscosity(
        air_temp_k)
    return 0.032 * re ** 0.85 * k_air / diameter_m


def surface_heat_balance(flux, diameter_m, speed, air_temp_k, air_density,
                         air_pressure=None, vapour_pressure=None):
    """Heat a 0 degC collector can lose, W per metre, before any freezing.

    ``Q_c + Q_e + Q_l - Q_v - Q_k`` of Makkonen (2000) for a collected water
    flux ``flux`` (kg m-1 s-1); the frozen fraction is this over
    ``flux L_f``, and a negative value is heat that melts ice.  Returns
    ``(balance, h)``.
    """

    xp = array_module(flux, diameter_m, speed, air_temp_k)
    tc = air_temp_k - KELVIN
    v = xp.maximum(speed, 0.0)
    h = heat_transfer_coefficient(diameter_m, v, air_temp_k, air_density)
    perimeter = math.pi * diameter_m
    balance = (h * perimeter * (-tc) + flux * CW * (-tc)
               - h * perimeter * RECOVERY_FACTOR * v ** 2 / (2.0 * CP_AIR)
               - flux * v ** 2 / 2.0)
    if air_pressure is not None:
        e_a = (saturation_vapour_pressure_water(tc) if vapour_pressure is None
               else vapour_pressure)
        balance = balance + h * perimeter * EPSILON * L_EVAP * (
            611.2 - e_a) / (CP_AIR * air_pressure)
    return balance, h


def freezing_fraction(balance, flux):
    """Makkonen ``a3`` from :func:`surface_heat_balance`; 1 with no flux."""

    xp = array_module(balance, flux)
    with np.errstate(divide="ignore", invalid="ignore"):
        n = xp.where(flux > 0, balance / (xp.where(flux > 0, flux, 1.0)
                                          * L_FUSION), 1.0)
    return xp.clip(n, 0.0, 1.0)


def blade_relative_speed(wind_speed: Any, blade: BladeProxy = DEFAULT_BLADE
                         ) -> tuple[Any, Any]:
    """Speed (m/s) the blade-proxy section sees, and the running mask."""

    xp = array_module(wind_speed)
    v = xp.maximum(xp.asarray(wind_speed, dtype=xp.float64), 0.0)
    running = (v >= blade.cut_in_m_s) & (v <= blade.cut_out_m_s)
    tip = xp.minimum(blade.tip_speed_ratio * v, blade.max_tip_speed_m_s)
    rotating = xp.hypot(blade.radial_fraction * tip,
                        (1.0 - blade.axial_induction) * v)
    speed = xp.where(running, rotating, v)
    return xp.where(xp.isfinite(v), speed, xp.nan), running


def iso12494_class(mass_kg_m: Any) -> Any:
    """ISO 12494 rime class (1-10), 0 below trace, -1 undefined (int8)."""

    xp = array_module(mass_kg_m)
    m = xp.asarray(mass_kg_m, dtype=xp.float64)
    cls = xp.ones(m.shape, dtype=xp.int8)
    for limit in ISO12494_RIME_LIMITS_KG_M:
        cls = cls + (m > limit).astype(xp.int8)
    cls = xp.where(m >= ISO12494_TRACE_KG_M, cls, 0)
    return xp.where(xp.isfinite(m), cls, -1).astype(xp.int8)


def iea_ice_class(met_icing_fraction: Any) -> Any:
    """IEA Wind Task 19 ice class 1-5 from the meteorological icing
    fraction (0-1) of the period; -1 where undefined (int8)."""

    xp = array_module(met_icing_fraction)
    pct = 100.0 * xp.asarray(met_icing_fraction, dtype=xp.float64)
    cls = xp.ones(pct.shape, dtype=xp.int8)
    for limit in IEA_MET_ICING_LIMITS_PCT:
        cls = cls + (pct > limit).astype(xp.int8)
    return xp.where(xp.isfinite(pct), cls, -1).astype(xp.int8)


def production_loss_fraction(blade_mass_kg_m: Any,
                             loss: IcingLossModel = DEFAULT_LOSS) -> Any:
    """Fraction of the un-iced power lost to blade ice (NaN unknown)."""

    xp = array_module(blade_mass_kg_m)
    m = xp.asarray(blade_mass_kg_m, dtype=xp.float64)
    out = xp.where(m < loss.trace_kg_m, 0.0,
                   xp.where(m >= loss.stop_kg_m, 1.0,
                            loss.operating_loss_max * m / loss.stop_kg_m))
    return xp.where(xp.isfinite(m), out, xp.nan)


def _flag(condition, defined):
    xp = array_module(condition)
    return xp.where(defined, condition.astype(xp.int8), -1).astype(xp.int8)


# --------------------------------------------------------------------------
# accretion over time


def accrete(*, time_seconds: Sequence[float], air_temp_k, speed, cloud_lwc,
            rain_lwc, mvd_m, air_density, diameter_m: float,
            air_pressure=None, vapour_pressure=None,
            initial_mass=None, initial_volume=None) -> dict[str, Any]:
    """Accrete and remove ice on one collector; arrays are (time, ...).

    ``speed`` is the speed the collector sees, ``cloud_lwc``/``rain_lwc``
    kg m-3 (``rain_lwc`` may be None), ``mvd_m`` the droplet MVD (NaN where
    there is no cloud), ``vapour_pressure`` the ambient ``e_a`` in Pa (None:
    saturated, and no sublimation).  Returns ``mass`` (kg/m), ``rate`` (the
    gross accretion rate, kg m-1 s-1), ``removal`` (melting plus
    sublimation actually applied, kg m-1 s-1), ``rime`` (int8 1 where the
    step grew dry), and the end state ``mass_end``/``volume_end`` (m3/m).
    """

    xp = array_module(air_temp_k, speed, cloud_lwc)
    seconds = np.asarray(time_seconds, dtype=np.float64)
    shape = air_temp_k.shape
    n_time, space = shape[0], shape[1:]
    if seconds.shape != (n_time,):
        raise TurbineIcingRefused(
            f"{seconds.size} time values for {n_time} time steps")
    if n_time > 1 and np.any(np.diff(seconds) <= 0):
        raise TurbineIcingRefused("time must be strictly increasing")
    zeros = xp.zeros(space, dtype=xp.float64)
    mass = zeros.copy() if initial_mass is None else \
        xp.broadcast_to(xp.asarray(initial_mass, dtype=xp.float64),
                        space).copy()
    if initial_volume is None:
        volume = mass / RIME_DENSITY
    else:
        volume = xp.broadcast_to(xp.asarray(initial_volume, dtype=xp.float64),
                                 space).copy()
    out_mass = xp.empty(shape, dtype=xp.float64)
    out_rate = xp.empty(shape, dtype=xp.float64)
    out_removal = xp.empty(shape, dtype=xp.float64)
    out_rime = xp.empty(shape, dtype=xp.int8)
    dead = ~(xp.isfinite(mass) & xp.isfinite(volume))
    for step in range(n_time):
        t = air_temp_k[step]
        tc = t - KELVIN
        v = xp.maximum(speed[step], 0.0)
        rho = air_density[step]
        diameter = xp.sqrt(diameter_m ** 2 + 4.0 * xp.where(
            dead, 0.0, volume) / math.pi)
        wc = xp.maximum(cloud_lwc[step], 0.0)
        mvd = xp.where(xp.isfinite(mvd_m[step]), mvd_m[step], 20e-6)
        a1 = collision_efficiency(diameter, v, t, rho, mvd)
        flux = a1 * wc * v * diameter
        if rain_lwc is not None:
            wr = xp.maximum(rain_lwc[step], 0.0)
            flux = flux + wr * xp.hypot(v, RAIN_FALL_SPEED) * diameter
        pressure = None if air_pressure is None else air_pressure[step]
        e_a = None if vapour_pressure is None else vapour_pressure[step]
        balance, h = surface_heat_balance(flux, diameter, v, t, rho,
                                          pressure, e_a)
        a3 = freezing_fraction(balance, flux)
        freezing = tc < 0.0
        rate = xp.where(freezing, a3 * flux, 0.0)
        dry = freezing & (a3 >= 1.0) & (flux > 0)
        melt = xp.maximum(-balance, 0.0) / L_FUSION
        removal = melt
        if pressure is not None and e_a is not None:
            sublimation = h * math.pi * diameter * EPSILON * xp.maximum(
                saturation_vapour_pressure_ice(xp.minimum(tc, 0.0)) - e_a,
                0.0) / (CP_AIR * pressure)
            # only a frozen surface sublimates: wet growth is at 0 degC
            # with its evaporation already in the heat balance
            frozen = balance >= flux * L_FUSION
            removal = removal + xp.where(frozen, sublimation, 0.0)
        bad = ~(xp.isfinite(rate) & xp.isfinite(removal) & xp.isfinite(t))
        dead = dead | bad
        rate = xp.where(dead, xp.nan, rate)
        if step:
            dt = float(seconds[step] - seconds[step - 1])
            gained = rate * dt
            grown = mass + gained
            new_mass = xp.maximum(grown - removal * dt, 0.0)
            applied = (grown - new_mass) / dt
            new_volume = volume + gained / xp.where(dry, RIME_DENSITY,
                                                    GLAZE_DENSITY)
            with np.errstate(divide="ignore", invalid="ignore"):
                keep = xp.where(grown > 0, new_mass / xp.where(
                    grown > 0, grown, 1.0), 0.0)
            mass = xp.where(dead, xp.nan, new_mass)
            volume = xp.where(dead, xp.nan, new_volume * keep)
        else:
            applied = zeros
        out_mass[step] = xp.where(dead, xp.nan, mass)
        out_rate[step] = rate
        out_removal[step] = xp.where(dead, xp.nan, applied)
        out_rime[step] = xp.where(dead, -1, dry.astype(xp.int8))
    return {"mass": out_mass, "rate": out_rate, "removal": out_removal,
            "rime": out_rime, "mass_end": mass, "volume_end": volume}


# --------------------------------------------------------------------------
# the kernel


def turbine_icing_arrays(*, time_seconds: Sequence[float], air_temperature,
                         wind_speed, air_density, cloud_mixing_ratio,
                         rain_mixing_ratio=None, cloud_number=None,
                         air_pressure=None, relative_humidity=None,
                         power_curve: Mapping[str, Any] | None = None,
                         blade: BladeProxy = DEFAULT_BLADE,
                         loss: IcingLossModel = DEFAULT_LOSS,
                         default_cloud_number_m3: float =
                         DEFAULT_CLOUD_NUMBER_M3,
                         initial_state: Mapping[str, Any] | None = None
                         ) -> dict[str, Any]:
    """Turbine icing from hub-height weather; arrays are (time, ...).

    Inputs: ``air_temperature`` K, ``wind_speed`` m/s, ``air_density``
    kg m-3, ``cloud_mixing_ratio``/``rain_mixing_ratio`` kg/kg,
    ``cloud_number`` droplets per kg of air (Thompson QNCLOUD; None uses
    ``default_cloud_number_m3``), ``air_pressure`` Pa and
    ``relative_humidity`` % (both optional).  ``power_curve`` is a
    :func:`woof.energy.products.load_power_curve` mapping (needed for the
    energy-weighted loss; without it that output is absent).
    ``initial_state`` holds ``reference_mass``, ``reference_volume``,
    ``blade_mass`` and ``blade_volume`` from the end of a previous chunk.

    Returns time series (``lwc``, ``mvd``, ``met_icing``, ``blade_speed``,
    ``reference_mass``, ``reference_rate``, ``instrumental_icing``,
    ``iso_class``, ``blade_mass``, ``blade_rate``, ``rotor_icing``,
    ``loss_fraction``) and per-point summaries (``met_icing_fraction``,
    ``instrumental_icing_fraction``, ``rotor_icing_fraction``,
    ``iea_class``, ``energy_loss_fraction``, the four end states), all
    with the array module of the inputs.
    """

    xp = array_module(air_temperature, wind_speed, air_density,
                      cloud_mixing_ratio)

    def arr(a):
        return None if a is None else xp.asarray(a, dtype=xp.float64)

    ta, ws, rho = arr(air_temperature), arr(wind_speed), arr(air_density)
    qc, qr = arr(cloud_mixing_ratio), arr(rain_mixing_ratio)
    nc, p, rh = arr(cloud_number), arr(air_pressure), arr(relative_humidity)
    shapes = {name: a.shape for name, a in (
        ("air_temperature", ta), ("wind_speed", ws), ("air_density", rho),
        ("cloud_mixing_ratio", qc), ("rain_mixing_ratio", qr),
        ("cloud_number", nc), ("air_pressure", p),
        ("relative_humidity", rh)) if a is not None}
    if len(set(shapes.values())) != 1 or ta.ndim < 1:
        raise TurbineIcingRefused(f"input shapes disagree: {shapes}")
    cloud_lwc = xp.maximum(qc, 0.0) * rho
    rain_lwc = None if qr is None else xp.maximum(qr, 0.0) * rho
    lwc = cloud_lwc if rain_lwc is None else cloud_lwc + rain_lwc
    default_number = xp.full(ta.shape, float(default_cloud_number_m3))
    if nc is None:
        number = default_number
    else:
        # a cloudy point without a usable droplet number (zero, or a
        # domain that did not write it) takes the documented default
        number = nc * rho
        number = xp.where(xp.isfinite(number) & (number > 0), number,
                          default_number)
    mvd = droplet_mvd(cloud_lwc, number)
    vapour = None
    if rh is not None:
        vapour = xp.clip(rh, 0.0, 100.0) / 100.0 * \
            saturation_vapour_pressure_water(ta - KELVIN)
    state = dict(initial_state or {})
    common = dict(time_seconds=time_seconds, air_temp_k=ta,
                  cloud_lwc=cloud_lwc, rain_lwc=rain_lwc, mvd_m=mvd,
                  air_density=rho, air_pressure=p, vapour_pressure=vapour)
    reference = accrete(speed=ws, diameter_m=REFERENCE_DIAMETER_M,
                        initial_mass=state.get("reference_mass"),
                        initial_volume=state.get("reference_volume"),
                        **common)
    blade_speed, running = blade_relative_speed(ws, blade)
    rotor = accrete(speed=blade_speed,
                    diameter_m=blade.collector_diameter_m,
                    initial_mass=state.get("blade_mass"),
                    initial_volume=state.get("blade_volume"), **common)
    met = (ta < KELVIN) & (lwc > MET_ICING_LWC_KG_M3)
    met_icing = _flag(met, xp.isfinite(ta) & xp.isfinite(lwc))
    instrumental = _flag(reference["mass"] >= ICE_PRESENT_KG_M,
                         xp.isfinite(reference["mass"]))
    rotor_icing = _flag(rotor["mass"] >= ICE_PRESENT_KG_M,
                        xp.isfinite(rotor["mass"]))
    loss_fraction = production_loss_fraction(rotor["mass"], loss)

    def fraction(flag):
        defined = (flag >= 0).sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            return xp.where(defined > 0, (flag == 1).sum(axis=0)
                            / xp.maximum(defined, 1), xp.nan)

    met_fraction = fraction(met_icing)
    out = {
        "lwc": lwc, "mvd": mvd, "met_icing": met_icing,
        "blade_speed": blade_speed,
        "reference_mass": reference["mass"],
        "reference_rate": reference["rate"],
        "instrumental_icing": instrumental,
        "iso_class": iso12494_class(reference["mass"]),
        "blade_mass": rotor["mass"], "blade_rate": rotor["rate"],
        "rotor_icing": rotor_icing, "loss_fraction": loss_fraction,
        "met_icing_fraction": met_fraction,
        "instrumental_icing_fraction": fraction(instrumental),
        "rotor_icing_fraction": fraction(rotor_icing),
        "iea_class": iea_ice_class(met_fraction),
        "reference_mass_end": reference["mass_end"],
        "reference_volume_end": reference["volume_end"],
        "blade_mass_end": rotor["mass_end"],
        "blade_volume_end": rotor["volume_end"],
    }
    if power_curve is not None:
        reference = power_curve.get("reference_density_kg_m3",
                                    power_curve.get("reference_density",
                                                    1.225))
        veq = ws * (rho / float(reference)) ** (1.0 / 3.0)
        speeds = xp.asarray(power_curve["speeds"], dtype=xp.float64)
        fractions = xp.asarray(power_curve["fraction"], dtype=xp.float64)
        free = xp.interp(veq.ravel(), speeds, fractions).reshape(veq.shape)
        free = xp.where((veq < power_curve["cut_in_ms"])
                        | (veq > power_curve["cut_out_ms"]), 0.0, free)
        valid = xp.isfinite(veq) & xp.isfinite(loss_fraction)
        free = xp.where(valid, free, 0.0)
        lost = (free * xp.where(valid, loss_fraction, 0.0)).sum(axis=0)
        total = free.sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            out["energy_loss_fraction"] = xp.where(
                total > 0, lost / xp.where(total > 0, total, 1.0), xp.nan)
    return out


#: Output variables: kernel key -> (name, units, long_name).  Time series
#: first, then the per-point summaries.
SERIES_OUTPUTS: dict[str, tuple[str, str | None, str]] = {
    "lwc": ("turbine_lwc", "kg m-3",
            "liquid water content at hub height (cloud + rain)"),
    "mvd": ("turbine_droplet_mvd", "m",
            "cloud droplet median volume diameter at hub height"),
    "met_icing": ("turbine_meteorological_icing", None,
                  "meteorological icing at hub height"),
    "blade_speed": ("turbine_blade_relative_speed", "m s-1",
                    "air speed seen by the blade-proxy section"),
    "reference_mass": ("turbine_reference_ice_mass", "kg m-1",
                       "ice on the ISO 12494 reference collector at hub "
                       "height"),
    "reference_rate": ("turbine_reference_icing_rate", "kg m-1 s-1",
                       "accretion rate on the reference collector"),
    "instrumental_icing": ("turbine_instrumental_icing", None,
                           "ice present on the reference collector"),
    "iso_class": ("turbine_iso12494_class", None,
                  "ISO 12494 rime class of the reference-collector ice"),
    "blade_mass": ("turbine_blade_ice_mass", "kg m-1",
                   "ice on the blade-proxy section"),
    "blade_rate": ("turbine_blade_icing_rate", "kg m-1 s-1",
                   "accretion rate on the blade-proxy section"),
    "rotor_icing": ("turbine_rotor_icing", None,
                    "ice present on the blade-proxy section"),
    "loss_fraction": ("turbine_icing_loss_fraction", "1",
                      "fraction of un-iced power lost to blade ice"),
}
SUMMARY_OUTPUTS: dict[str, tuple[str, str | None, str]] = {
    "met_icing_fraction": ("turbine_meteorological_icing_fraction", "1",
                           "fraction of the period with meteorological "
                           "icing"),
    "instrumental_icing_fraction": ("turbine_instrumental_icing_fraction",
                                    "1", "fraction of the period with "
                                    "instrumental icing"),
    "rotor_icing_fraction": ("turbine_rotor_icing_fraction", "1",
                             "fraction of the period with rotor icing"),
    "iea_class": ("turbine_iea_ice_class", None,
                  "IEA Wind Task 19 ice class of the period"),
    "energy_loss_fraction": ("turbine_icing_energy_loss_fraction", "1",
                             "fraction of the period's un-iced energy lost "
                             "to blade ice"),
}
#: End-of-period state for chunked hindcasts (``initial_state`` keys).
STATE_OUTPUTS: dict[str, tuple[str, str, str]] = {
    "reference_mass_end": ("turbine_reference_ice_mass_end", "kg m-1",
                           "reference-collector ice at the last time"),
    "reference_volume_end": ("turbine_reference_ice_volume_end", "m3 m-1",
                             "reference-collector ice volume at the last "
                             "time"),
    "blade_mass_end": ("turbine_blade_ice_mass_end", "kg m-1",
                       "blade-proxy ice at the last time"),
    "blade_volume_end": ("turbine_blade_ice_volume_end", "m3 m-1",
                         "blade-proxy ice volume at the last time"),
}
FLAG_MEANINGS = "undefined no yes"


def variable_attrs(key: str, *, blade: BladeProxy = DEFAULT_BLADE,
                   loss: IcingLossModel = DEFAULT_LOSS,
                   droplet_number: str = "") -> dict[str, Any]:
    """CF-style attributes, with the method and assumptions, for one key."""

    table = {**SERIES_OUTPUTS, **SUMMARY_OUTPUTS, **STATE_OUTPUTS}
    _, units, long_name = table[key]
    attrs: dict[str, Any] = {"long_name": long_name}
    if units:
        attrs["units"] = units
    accretion = (
        "Makkonen (2000) dM/dt = a1 a2 a3 w v D, a2 = 1; a1 Finstad et al. "
        "(1988) at the droplet MVD for cloud water, 1 for rain at "
        f"sqrt(v^2 + {RAIN_FALL_SPEED:g}^2) m/s; a3 the frozen fraction of "
        "the Makkonen heat balance (convective, evaporative, droplet "
        "warming, aerodynamic heating r = 0.79, droplet kinetic energy); "
        f"rime {RIME_DENSITY:g} / glaze {GLAZE_DENSITY:g} kg m-3; removal "
        "by melting (heat-balance surplus) and sublimation (with pressure "
        "and humidity); no shedding")
    if key.endswith("_end"):
        attrs["method"] = ("pass the output back as initial_state to "
                           "continue a chunked hindcast")
    elif key == "lwc":
        attrs["method"] = "rho (q_cloud + q_rain)"
    elif key == "mvd":
        attrs["method"] = (
            "Thompson gamma spectrum, mu = clip(nint(1e9/N_c) + 2, 2, 15), "
            "MVD = (mu + 3.67) / lambda, clipped to "
            f"[{MVD_MIN_M * 1e6:g}, {MVD_MAX_M * 1e6:g}] um; NaN without "
            "cloud water")
        if droplet_number:
            attrs["droplet_number"] = droplet_number
    elif key == "met_icing":
        attrs["method"] = (f"air temperature < 0 degC and LWC > "
                           f"{MET_ICING_LWC_KG_M3 * 1e3:g} g m-3 "
                           "(IEA Wind Task 19 meteorological icing)")
    elif key == "blade_speed":
        attrs["method"] = blade.describe()
    elif key.startswith("reference"):
        attrs["method"] = (f"{accretion}; {REFERENCE_DIAMETER_M * 1e3:g} mm "
                           "ISO 12494 reference collector in the hub wind")
    elif key == "instrumental_icing":
        attrs["method"] = (f"reference-collector ice >= {ICE_PRESENT_KG_M:g} "
                           "kg/m (IEA Wind Task 19 instrumental icing)")
    elif key == "iso_class":
        attrs["thresholds_kg_m"] = np.array(ISO12494_RIME_LIMITS_KG_M)
        attrs["method"] = ("ISO 12494 Table 4 rime classes on the "
                           "reference-collector ice at hub height")
    elif key.startswith("blade"):
        attrs["method"] = f"{accretion}; {blade.describe()}"
    elif key == "rotor_icing":
        attrs["method"] = (f"blade-proxy ice >= {ICE_PRESENT_KG_M:g} kg/m")
    elif key == "loss_fraction":
        attrs["method"] = loss.describe()
        attrs["assumptions"] = ("uncalibrated: set IcingLossModel from SCADA; "
                                "no feedback of a stopped rotor on accretion")
    elif key == "iea_class":
        attrs["method"] = (
            "IEA Wind Task 19 classes 1-5 by the meteorological icing "
            "fraction, upper bounds "
            f"{', '.join(f'{x:g}' for x in IEA_MET_ICING_LIMITS_PCT)} % "
            "(class 5 above); defined on a year, applied to the period")
    elif key == "energy_loss_fraction":
        attrs["method"] = ("sum(P L) / sum(P), P the power curve at the "
                           "density-corrected hub wind; NaN with no "
                           "un-iced production")
    elif key.endswith("_fraction"):
        attrs["method"] = "share of the defined output times flagged yes"
    if key in ("met_icing", "instrumental_icing", "rotor_icing"):
        attrs["flag_values"] = np.array([-1, 0, 1], dtype=np.int8)
        attrs["flag_meanings"] = FLAG_MEANINGS
    elif key == "iso_class":
        attrs["flag_values"] = np.arange(-1, 11, dtype=np.int8)
        attrs["flag_meanings"] = "undefined none " + " ".join(
            f"R{i}" for i in range(1, 11))
    elif key == "iea_class":
        attrs["flag_values"] = np.arange(-1, 6, dtype=np.int8)
        attrs["flag_meanings"] = "undefined " + " ".join(
            f"IEA{i}" for i in range(1, 6))
    return attrs


# --------------------------------------------------------------------------
# site series: ``woof energy rating --products turbine-icing``


def compute_turbine_icing(forecast, curve: Mapping[str, Any], notes
                          ) -> list:
    """``turbine-icing`` outputs for a :class:`products.Forecast`.

    Turbines (and plants with a hub height) only; every other site is NaN
    (flags -1).  Weather is read at ``hub_height_m``: wind linear in
    ln(height), everything else linear in height, the nearest height
    outside the forecast range (noted).
    """

    from woof.energy import products as P

    v = forecast.variables
    kind = forecast.coord("kind").astype(str)
    hub = P._coord_or_nan(forecast, "hub_height_m")
    mask = P._wind_mask(kind, hub)
    no_hub = (kind == "turbine") & ~np.isfinite(hub)
    if no_hub.any():
        notes.add(f"turbine-icing: {int(no_hub.sum())} turbine site(s) have "
                  "no hub_height_m; their icing is undefined")
    idx = np.flatnonzero(mask & np.isfinite(hub))
    n_time, n_site = len(forecast.times), forecast.n_site
    blade = BladeProxy(cut_in_m_s=float(curve["cut_in_ms"]),
                       cut_out_m_s=float(curve["cut_out_ms"]))
    result: dict[str, Any] = {}
    droplet_number = ""
    if idx.size:
        heights, at = forecast.heights, hub[idx]

        def hub_field(name, log=False):
            if name not in v:
                return None
            values, outside = P.interp_height(v[name][:, idx], heights, at,
                                              log=log)
            if outside.any():
                notes.add(f"turbine-icing: hub height outside the forecast "
                          f"heights for {int(outside.sum())} site(s); the "
                          "nearest height was used")
            return values

        if "cloud_number_mixing_ratio" in v:
            droplet_number = ("rho * cloud_number_mixing_ratio (Thompson "
                              "QNCLOUD)")
        else:
            droplet_number = (f"default N_c = {DEFAULT_CLOUD_NUMBER_M3:g} "
                              "m-3 (Thompson mp_physics=8 fixed value); the "
                              "forecast has no cloud_number_mixing_ratio")
            notes.add(f"turbine-icing: droplet MVD from the {droplet_number}")
        if "rain_mixing_ratio" not in v:
            notes.add("turbine-icing: no rain_mixing_ratio in the forecast; "
                      "LWC is cloud water only and freezing rain is not "
                      "accreted")
        if "air_pressure" not in v:
            notes.add("turbine-icing: no air_pressure in the forecast; "
                      "evaporative cooling and sublimation are left out")
        elif "relative_humidity" not in v:
            notes.add("turbine-icing: no relative_humidity in the forecast; "
                      "air is taken as saturated and ice does not sublimate")
        temperature = hub_field("air_temperature")
        density = hub_field("air_density")
        if density is None:
            density = hub_field("air_pressure") / (P._R_DRY * temperature)
            notes.add("turbine-icing: no air_density in the forecast; "
                      "density from air_pressure / (R_d air_temperature), "
                      "dry air")
        if "cloud_number_mixing_ratio" in v:
            unusable = ~(v["cloud_number_mixing_ratio"][:, idx] > 0) & \
                (v["cloud_liquid_mixing_ratio"][:, idx] > 0)
            if unusable.any():
                notes.add("turbine-icing: cloud water without a positive "
                          "droplet number at some heights and times; the "
                          f"default N_c = {DEFAULT_CLOUD_NUMBER_M3:g} m-3 "
                          "is used there")
        result = turbine_icing_arrays(
            time_seconds=forecast.time_seconds,
            air_temperature=temperature,
            wind_speed=hub_field("wind_speed", log=True),
            air_density=density,
            cloud_mixing_ratio=hub_field("cloud_liquid_mixing_ratio"),
            rain_mixing_ratio=hub_field("rain_mixing_ratio"),
            cloud_number=hub_field("cloud_number_mixing_ratio"),
            air_pressure=hub_field("air_pressure"),
            relative_humidity=hub_field("relative_humidity"),
            power_curve=curve, blade=blade)
        notes.add(f"turbine-icing: {blade.describe()}; no shedding, so rotor "
                  "icing duration is an upper bound")
        if n_time > 1:
            span = forecast.time_seconds[-1] - forecast.time_seconds[0]
            if span < 360.0 * 86400.0:
                notes.add("turbine-icing: the IEA ice class is defined on a "
                          "year; this period is shorter and is classed as "
                          "if it were one")
    outputs = []
    for key in (*SERIES_OUTPUTS, *SUMMARY_OUTPUTS):
        name = {**SERIES_OUTPUTS, **SUMMARY_OUTPUTS}[key][0]
        series = key in SERIES_OUTPUTS
        flag = key in ("met_icing", "instrumental_icing", "rotor_icing",
                       "iso_class", "iea_class")
        shape = (n_time, n_site) if series else (n_site,)
        data = np.full(shape, -1, dtype=np.int8) if flag else \
            np.full(shape, np.nan)
        if key in result:
            values = _to_numpy(result[key])
            if series:
                data[:, idx] = values
            else:
                data[idx] = values
        attrs = variable_attrs(key, blade=blade,
                               droplet_number=droplet_number)
        if key == "energy_loss_fraction":
            attrs["power_curve"] = curve["key"]
            attrs["power_curve_sha256"] = curve["sha256"]
        outputs.append(P.ProductOutput(
            name, ("time", "site") if series else ("site",), data, attrs))
    return outputs


# --------------------------------------------------------------------------
# gridded: wrfout / Icechunk datasets


#: Kernel inputs in apply_ufunc order: forecast-variable name -> kernel arg.
GRID_INPUTS = (
    ("air_temperature", "air_temperature"),
    ("wind_speed", "wind_speed"),
    ("air_density", "air_density"),
    ("cloud_liquid_mixing_ratio", "cloud_mixing_ratio"),
    ("rain_mixing_ratio", "rain_mixing_ratio"),
    ("cloud_number_mixing_ratio", "cloud_number"),
    ("air_pressure", "air_pressure"),
    ("relative_humidity", "relative_humidity"),
)
#: wrfout sources the hub-height interpolation needs, and optional ones.
WRFOUT_REQUIRED = ("PH", "PHB", "HGT", "T", "P", "PB", "QVAPOR", "QCLOUD",
                   "U", "V")
WRFOUT_OPTIONAL = ("QRAIN", "QNCLOUD")


def interp_columns(field, z, target: float, axis: int, *, log: bool = False):
    """Linear (or ln-height) interpolation of ``field`` to height ``target``
    along ``axis`` where ``z`` (same shape) increases; NaN outside."""

    xp = array_module(field, z)
    field = xp.moveaxis(xp.asarray(field, dtype=xp.float64), axis, -1)
    z = xp.moveaxis(xp.asarray(z, dtype=xp.float64), axis, -1)
    nz = z.shape[-1]
    if nz < 2:
        raise TurbineIcingRefused("vertical interpolation needs at least two "
                                  "levels")
    hi = xp.clip((z < target).sum(axis=-1, keepdims=True), 1, nz - 1)
    lo = hi - 1
    z_lo = xp.take_along_axis(z, lo, axis=-1)[..., 0]
    z_hi = xp.take_along_axis(z, hi, axis=-1)[..., 0]
    f_lo = xp.take_along_axis(field, lo, axis=-1)[..., 0]
    f_hi = xp.take_along_axis(field, hi, axis=-1)[..., 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        if log:
            weight = (math.log(target) - xp.log(z_lo)) / (xp.log(z_hi)
                                                         - xp.log(z_lo))
        else:
            weight = (target - z_lo) / (z_hi - z_lo)
    out = f_lo + (f_hi - f_lo) * weight
    inside = (z[..., 0] <= target) & (z[..., -1] >= target)
    return xp.where(inside, out, xp.nan)


def _time_dim(ds, time_dim: str | None) -> str:
    if time_dim is not None:
        if time_dim not in ds.dims:
            raise TurbineIcingRefused(f"no {time_dim!r} dimension in the "
                                      "dataset")
        return time_dim
    for name in ("time", "Time", "valid_time"):
        if name in ds.dims:
            return name
    raise TurbineIcingRefused(
        f"no time dimension (time, Time or valid_time) among {list(ds.dims)}"
        "; pass time_dim")


def _valid_times(ds, time_dim: str) -> tuple[np.ndarray, np.ndarray]:
    """``(seconds since the first time, datetime64[s] times)`` from a
    datetime64 coordinate, XTIME or the wrfout ``Times`` strings."""

    candidates = [ds[time_dim]] if time_dim in ds.coords else []
    if "XTIME" in ds.variables:
        candidates.append(ds["XTIME"])
    for candidate in candidates:
        values = np.asarray(candidate.values)
        if values.dtype.kind == "M" and values.ndim == 1:
            values = values.astype("datetime64[s]")
            return (values - values[0]) / np.timedelta64(1, "s"), values
    if "Times" in ds.variables:
        raw = np.asarray(ds["Times"].values)
        texts = []
        for row in raw:
            if isinstance(row, (bytes, np.bytes_)):
                text = row.decode("ascii", "replace")
            elif np.ndim(row):
                text = b"".join(bytes(c) for c in row).decode("ascii",
                                                              "replace")
            else:
                text = str(row)
            texts.append(np.datetime64(text.strip("\x00 ").replace("_", "T"),
                                       "s"))
        values = np.array(texts, dtype="datetime64[s]")
        return (values - values[0]) / np.timedelta64(1, "s"), values
    raise TurbineIcingRefused(
        f"cannot read valid times: {time_dim!r} is not a datetime64 "
        "coordinate and there is no XTIME or Times variable")


def wrfout_hub_fields(ds, hub_height_m: float) -> dict[str, Any]:
    """Forecast-variable fields at ``hub_height_m`` above model terrain from
    a wrfout dataset (xarray, dims ``Time, bottom_top, south_north,
    west_east`` and the staggered ones).

    Heights are ``(PH + PHB) / g - HGT`` at mass levels (destaggered), as
    the site sampler measures them; U and V are destaggered to mass points
    (speed does not depend on the grid rotation); wind is interpolated
    linearly in ln(height), everything else linearly in height; NaN outside
    a column.  ``T = (T + 300)(p / p0)^(R_d / c_p)``, ``rho = p / (R_d T
    (1 + 0.608 q))``, relative humidity over liquid water (Bolton) -- the
    :mod:`woof.energy.extract` derivations.
    """

    import xarray as xr

    from woof.core.constants import G
    from woof.energy import extract as E

    missing = [name for name in WRFOUT_REQUIRED if name not in ds.variables]
    if missing:
        raise TurbineIcingRefused(
            f"wrfout dataset lacks {', '.join(missing)} for hub-height "
            "interpolation")
    if not (isinstance(hub_height_m, numbers.Real)
            and math.isfinite(hub_height_m) and hub_height_m > 0):
        raise TurbineIcingRefused("hub_height_m must be a positive number of "
                                  "metres above ground")

    def var(name):
        return ds[name].variable

    def destagger(variable, dim):
        n = variable.sizes[dim]
        mean = 0.5 * (variable.isel({dim: slice(0, n - 1)})
                      + variable.isel({dim: slice(1, n)}))
        return xr.Variable(tuple(d.replace("_stag", "") for d in mean.dims),
                           mean.data)

    z = destagger((var("PH") + var("PHB")) / G, "bottom_top_stag") - var("HGT")
    pressure = var("P") + var("PB")
    temperature = E.air_temperature(var("T") + 300.0, pressure)
    qv = var("QVAPOR")
    columns = {
        "air_temperature": temperature,
        "air_pressure": pressure,
        "air_density": E.air_density(pressure, temperature, qv),
        "relative_humidity": (100.0 * (qv * pressure / (E.EP2 + qv))
                              / E.saturation_vapour_pressure(temperature)
                              ).clip(0.0, 100.0),
        "cloud_liquid_mixing_ratio": var("QCLOUD"),
        "wind_speed": np.hypot(destagger(var("U"), "west_east_stag"),
                               destagger(var("V"), "south_north_stag")),
    }
    if "QRAIN" in ds.variables:
        columns["rain_mixing_ratio"] = var("QRAIN")
    if "QNCLOUD" in ds.variables:
        columns["cloud_number_mixing_ratio"] = var("QNCLOUD")
    out = {}
    for name, column in columns.items():
        column = xr.DataArray(column)
        out[name] = xr.apply_ufunc(
            interp_columns, column, xr.DataArray(z),
            input_core_dims=[["bottom_top"], ["bottom_top"]],
            kwargs=dict(target=float(hub_height_m), axis=-1,
                        log=name == "wind_speed"),
            dask="parallelized", output_dtypes=[np.float64],
            dask_gufunc_kwargs={"allow_rechunk": True})
    return out


def height_hub_fields(ds, hub_height_m: float | None, names: Sequence[str],
                      notes: list[str] | None = None) -> dict[str, Any]:
    """Forecast-variable fields at ``hub_height_m`` from a dataset with a 1-D
    ``height`` coordinate (or already at hub height when there is none).

    Wind speed (or the speed of ``u``/``v``) is interpolated linearly in
    ln(height), everything else linearly in height, the nearest height
    outside the range, and linearly where the lower level is not above
    ground.  Without ``air_density``, ``p / (R_d T)``.  Approximations are
    appended to ``notes``.
    """

    notes = [] if notes is None else notes
    has_height = "height" in ds.dims
    if has_height and hub_height_m is None:
        raise TurbineIcingRefused("the dataset has a height dimension; pass "
                                  "hub_height_m")
    if has_height:
        heights = np.asarray(ds["height"].values, dtype=np.float64)
        if heights.size > 1 and np.any(np.diff(heights) <= 0):
            raise TurbineIcingRefused("height must be strictly increasing")
        target = float(np.clip(hub_height_m, heights[0], heights[-1]))
        if target != float(hub_height_m):
            notes.append(f"hub height {float(hub_height_m):g} m is outside "
                         f"the dataset heights {heights[0]:g}-"
                         f"{heights[-1]:g} m; the nearest height was used")
        hi = int(np.clip(np.searchsorted(heights, target), 1,
                         max(heights.size - 1, 1)))
        lo = hi - 1

    def at_hub(field, log: bool):
        if not has_height or "height" not in field.dims:
            return field
        if heights.size == 1:
            return field.isel(height=0, drop=True)
        z_lo, z_hi = heights[lo], heights[hi]
        if log and z_lo <= 0.0:
            notes.append(f"wind interpolated linearly in height: the level "
                         f"below the hub is at {z_lo:g} m")
            log = False
        if log:
            weight = math.log(target / z_lo) / math.log(z_hi / z_lo)
        else:
            weight = (target - z_lo) / (z_hi - z_lo)
        below = field.isel(height=lo, drop=True)
        above = field.isel(height=hi, drop=True)
        return below + (above - below) * weight

    if not has_height and hub_height_m is not None:
        notes.append(f"the dataset has no height dimension; its fields are "
                     f"taken as at the {float(hub_height_m):g} m hub")
    out = {}
    for name in names:
        if name in ds.variables:
            out[name] = at_hub(ds[name], name == "wind_speed")
    if "wind_speed" not in out and {"u", "v"} <= set(ds.variables):
        out["wind_speed"] = at_hub(np.hypot(ds["u"], ds["v"]), True)
    if "air_density" not in out and {"air_pressure", "air_temperature"} <= \
            set(out):
        from woof.energy.products import _R_DRY

        out["air_density"] = out["air_pressure"] / (_R_DRY
                                                    * out["air_temperature"])
    return out


def turbine_icing_dataset(ds, *, hub_height_m: float | None = None,
                          time_dim: str | None = None,
                          blade: BladeProxy | None = None,
                          loss: IcingLossModel = DEFAULT_LOSS,
                          power_curve: Mapping[str, Any] | str | None = None,
                          default_cloud_number_m3: float =
                          DEFAULT_CLOUD_NUMBER_M3,
                          include_rain: bool = True,
                          device: str = "auto",
                          initial_state=None):
    """Turbine icing on a grid: an xarray Dataset in, a Dataset out.

    ``ds`` is either a wrfout history (``PH``, ``PHB``, ``HGT``, ``T``,
    ``P``, ``PB``, ``QVAPOR``, ``QCLOUD``, ``U``, ``V``; ``QRAIN`` and
    Thompson's ``QNCLOUD`` used when present), interpolated to
    ``hub_height_m`` above model terrain by :func:`wrfout_hub_fields`; or a
    dataset carrying the ``forecast.v1`` variable names (``air_temperature``,
    ``wind_speed`` or ``u``/``v``, ``air_density`` or ``air_pressure``,
    ``cloud_liquid_mixing_ratio``; optional ``rain_mixing_ratio``,
    ``cloud_number_mixing_ratio``, ``air_pressure``, ``relative_humidity``)
    either on a 1-D ``height`` dimension (interpolated to ``hub_height_m``)
    or already at hub height.  Any other dimensions are kept.

    ``power_curve`` is a curve mapping or name in the packaged table
    (default: the generic curve).  ``device`` is ``"cpu"``, ``"gpu"``
    (CuPy; each chunk is moved to the GPU and back) or ``"auto"`` (CuPy if
    the data already are CuPy arrays).  ``initial_state`` is a previous
    output (its ``*_end`` variables) to continue a chunked hindcast.

    The result has the :data:`SERIES_OUTPUTS` on (time, ...), the
    :data:`SUMMARY_OUTPUTS` and :data:`STATE_OUTPUTS` on (...), attributes
    naming every assumption, and ``schema`` :data:`TURBINE_ICING_SCHEMA`.
    Dask-backed inputs stay lazy, mapped over spatial chunks.
    """

    import xarray as xr

    from woof.energy import products as P

    if device not in ("auto", "cpu", "gpu"):
        raise TurbineIcingRefused(f"device must be auto, cpu or gpu, not "
                                  f"{device!r}")
    if power_curve is None or isinstance(power_curve, str):
        curve = P.load_power_curve(name=power_curve)
    else:
        curve = dict(power_curve)
        curve.setdefault("key", "custom")
    blade = blade or BladeProxy(cut_in_m_s=float(curve["cut_in_ms"]),
                                cut_out_m_s=float(curve["cut_out_ms"]))
    time_dim = _time_dim(ds, time_dim)
    notes: list[str] = []
    wrfout = all(name in ds.variables for name in ("PH", "PHB", "HGT"))
    if wrfout:
        if hub_height_m is None:
            raise TurbineIcingRefused("a wrfout dataset needs hub_height_m")
        fields = wrfout_hub_fields(ds, hub_height_m)
        source = "wrfout model levels"
    else:
        fields = height_hub_fields(ds, hub_height_m,
                                   [name for name, _ in GRID_INPUTS], notes)
        source = ("forecast.v1 names on height levels" if "height" in ds.dims
                  else "forecast.v1 names at hub height")
    need = ("air_temperature", "wind_speed", "air_density",
            "cloud_liquid_mixing_ratio")
    missing = [name for name in need if name not in fields]
    if missing:
        raise TurbineIcingRefused(
            f"the dataset cannot supply {', '.join(missing)} at hub height")
    if not include_rain:
        fields.pop("rain_mixing_ratio", None)
    for name, field in fields.items():
        if time_dim not in field.dims:
            raise TurbineIcingRefused(f"{name} has no {time_dim!r} dimension")
    if "cloud_number_mixing_ratio" in fields:
        droplet_number = "rho * cloud droplet number (Thompson QNCLOUD)"
    else:
        droplet_number = (f"default N_c = {default_cloud_number_m3:g} m-3 "
                          "(Thompson mp_physics=8 fixed value)")
        notes.append(f"droplet MVD from the {droplet_number}")
    if "air_pressure" not in fields:
        notes.append("no air pressure: evaporative cooling and sublimation "
                     "left out")
    elif "relative_humidity" not in fields:
        notes.append("no relative humidity: air taken as saturated, no "
                     "sublimation")
    seconds, valid_times = _valid_times(ds, time_dim)
    if seconds.size != ds.sizes[time_dim]:
        raise TurbineIcingRefused(
            f"{seconds.size} valid times for {ds.sizes[time_dim]} records")
    present = [(name, arg) for name, arg in GRID_INPUTS if name in fields]
    state_names = []
    state_args = []
    if initial_state is not None:
        for key in STATE_OUTPUTS:
            name = STATE_OUTPUTS[key][0]
            if name not in initial_state.variables:
                raise TurbineIcingRefused(f"initial_state lacks {name}")
            state_names.append(key)
            state_args.append(initial_state[name])
    series_keys = list(SERIES_OUTPUTS)
    summary_keys = [*SUMMARY_OUTPUTS, *STATE_OUTPUTS]

    def kernel(*arrays):
        if device == "gpu":
            import cupy as xp
        elif device == "cpu":
            xp = np
            arrays = [_to_numpy(a) for a in arrays]
        else:
            xp = array_module(*arrays)
        series_in = [xp.asarray(a, dtype=xp.float64)
                     for a in arrays[:len(present)]]
        state_in = [xp.asarray(a, dtype=xp.float64)
                    for a in arrays[len(present):]]
        n_time = series_in[0].shape[-1]
        space = np.broadcast_shapes(*(a.shape[:-1] for a in series_in),
                                    *(a.shape for a in state_in))
        inputs = {arg: xp.moveaxis(xp.broadcast_to(a, space + (n_time,)),
                                   -1, 0)
                  for (_, arg), a in zip(present, series_in)}
        state = {key.replace("_end", ""): xp.broadcast_to(a, space)
                 for key, a in zip(state_names, state_in)}
        result = turbine_icing_arrays(
            time_seconds=seconds, power_curve=curve, blade=blade, loss=loss,
            default_cloud_number_m3=default_cloud_number_m3,
            initial_state=state or None, **inputs)
        series = [np.moveaxis(_to_numpy(result[k]), 0, -1).astype(np.float64)
                  for k in series_keys]
        summary = [_to_numpy(result[k]).astype(np.float64)
                   for k in summary_keys]
        return (*series, *summary)

    args = [fields[name] for name, _ in present] + state_args
    results = xr.apply_ufunc(
        kernel, *args,
        input_core_dims=[[time_dim]] * len(present) + [[]] * len(state_args),
        output_core_dims=[[time_dim]] * len(series_keys)
        + [[]] * len(summary_keys),
        dask="parallelized",
        output_dtypes=[np.float64] * (len(series_keys) + len(summary_keys)),
        dask_gufunc_kwargs={"allow_rechunk": True})
    table = {**SERIES_OUTPUTS, **SUMMARY_OUTPUTS, **STATE_OUTPUTS}
    flags = ("met_icing", "instrumental_icing", "rotor_icing", "iso_class",
             "iea_class")
    data_vars = {}
    for key, value in zip([*series_keys, *summary_keys], results):
        if key in flags:
            value = value.fillna(-1).astype(np.int8)
        value.attrs = variable_attrs(key, blade=blade, loss=loss,
                                     droplet_number=droplet_number)
        if key == "energy_loss_fraction":
            value.attrs["power_curve"] = curve.get("key", "custom")
        data_vars[table[key][0]] = value
    out = xr.Dataset(data_vars).transpose(time_dim, ...)
    out = out.assign_coords({time_dim: (time_dim, valid_times)})
    for name in ("XLAT", "XLONG", "lat", "lon", "latitude", "longitude"):
        if name in ds.variables:
            coord = ds[name]
            if time_dim in coord.dims and name in ("XLAT", "XLONG"):
                coord = coord.isel({time_dim: 0}, drop=True)
            if set(coord.dims) <= set(out.dims):
                out = out.assign_coords({name: coord.variable})
    out.attrs = {
        "schema": TURBINE_ICING_SCHEMA,
        "title": "woof turbine icing at hub height",
        "hub_height_m": float(hub_height_m) if hub_height_m is not None
        else "input fields are at hub height",
        "source_fields": source,
        "time_dim": time_dim,
        "power_curve": str(curve.get("key", "custom")),
        "blade_proxy": blade.describe(),
        "loss_model": loss.describe(),
        "notes": "\n".join(notes),
    }
    if curve.get("sha256"):
        out.attrs["power_curve_sha256"] = curve["sha256"]
    return out


__all__ = [
    "TURBINE_ICING_SCHEMA", "PRODUCT", "TurbineIcingRefused", "BladeProxy",
    "IcingLossModel", "DEFAULT_BLADE", "DEFAULT_LOSS", "array_module",
    "thompson_cloud_shape", "droplet_mvd", "collision_efficiency",
    "saturation_vapour_pressure_water", "saturation_vapour_pressure_ice",
    "heat_transfer_coefficient", "surface_heat_balance", "freezing_fraction",
    "blade_relative_speed", "iso12494_class", "iea_ice_class",
    "production_loss_fraction", "accrete", "turbine_icing_arrays",
    "SERIES_OUTPUTS", "SUMMARY_OUTPUTS", "STATE_OUTPUTS", "variable_attrs",
    "compute_turbine_icing", "interp_columns", "wrfout_hub_fields",
    "height_hub_fields", "turbine_icing_dataset",
]
