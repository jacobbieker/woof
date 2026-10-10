"""Energy products from a ``forecast.v1`` file: ``woof energy rating``.

Four products are derived from the per-site forecast time series written by
``woof energy extract``:

``dlr``
    Dynamic line rating of overhead conductors (sites of kind
    ``line_sample`` and ``tower``) by the IEEE Std 738-2012 steady-state heat
    balance ``I = sqrt((q_c + q_r - q_s) / R(T_max))``:

    * forced convection ``q_c1``/``q_c2`` with the wind-direction factor
      ``K_angle`` from ``wind_attack_angle`` (IEEE: phi is the angle between
      the wind and the conductor axis), natural convection ``q_cn``, and the
      largest of the three used;
    * radiated cooling ``q_r``;
    * solar gain ``q_s``: when the forecast carries ``ghi``, ``dni`` and
      ``dhi`` the conductor is heated by the forecast irradiance with the
      CIGRE TB 601 cylinder form (beam on the projected area at the beam
      incidence angle, isotropic sky diffuse and ground-reflected radiation,
      each times pi/2 for a horizontal cylinder, albedo 0.2).  With only
      ``ghi`` the irradiance is split by the Erbs decomposition first.  With
      no irradiance at all the IEEE 738 clear-atmosphere model is used, which
      is the conservative (cloud-free) choice.  The variable attribute
      ``solar_method`` and the global notes say which;
    * conductor resistance at ``T_max`` by linear interpolation between
      R(25 degC) and R(75 degC);
    * air density, viscosity and conductivity by the IEEE formulas at the
      film temperature and the elevation ``terrain_height + height_m``.

    Weather is read at the conductor height (``height_m`` of the conductor
    entry, 10 m by default) by linear interpolation between forecast heights;
    outside the forecast height range the nearest height is used and noted.
    ``rating_amps`` is per phase bundle (subconductor rating times
    ``subconductors``); ``rating_mva = sqrt(3) V I / 1e6`` for one circuit,
    because ``forecast.v1`` does not say how many circuits a line carries.

``icing``
    Makkonen (2000) accretion ``dM/dt = a1 a2 a3 w v D`` on the conductor (or
    the ISO 12494 30 mm reference collector at 10 m for sites that are not
    conductors), with ``w`` the cloud liquid water content
    ``cloud_liquid_mixing_ratio * air_density``, ``v`` the wind component
    normal to the conductor (the full speed where the axis is unknown),
    ``a1`` from Finstad et al. (1988) with an assumed median volume droplet
    diameter of 20 um, ``a2 = 1`` and ``a3`` = 1 for dry growth at or below
    -5 degC and otherwise the frozen fraction of a simplified Makkonen heat
    balance (convective and evaporative cooling plus the warming of the
    supercooled water to 0 degC against the latent heat of freezing; kinetic
    and radiative terms neglected).  A freezing-rain term follows Jones
    (1998): rain water ``rain_mixing_ratio * air_density`` falling at an
    assumed 4 m/s and driven by the normal wind, all of it freezing, when the
    air is below 0 degC.  The ice is accumulated over the forecast with no
    shedding and the collector diameter grows with the ice.  Rime is stored
    at 500 kg/m3 and glaze (wet growth and freezing rain) at 900 kg/m3 for the
    radial thickness.  ``icing_class`` is the ISO 12494 rime class R1-R10 of
    the accumulated mass per metre (0 = no ice).

``wind-power``
    Turbines (kind ``turbine``, and ``plant`` sites that carry a hub height):
    hub-height wind by log-linear interpolation between the bracketing
    forecast heights, the IEC 61400-12 density correction
    ``v (rho / 1.225)^(1/3)``, the generic normalised IEC class II curve from
    ``woof/data/energy/power_curves.json`` and ``capacity_mw``.

``pv-power``
    PV sites (kind ``pv``): a PVWatts-like model with a fixed tilt of
    ``0.76 |lat| + 3.1`` degrees facing the equator, plane-of-array
    irradiance by the isotropic (Liu-Jordan) sky model with albedo 0.2, the
    Faiman cell temperature from ``t2`` and ``wind_speed_10m``, a
    -0.4 %/K power temperature coefficient, 14 % system losses and a 96 %
    inverter efficiency.  ``capacity_mw`` is taken as the DC nameplate and AC
    output is clipped at it.  Without ``dni``/``dhi`` the Erbs decomposition
    of ``ghi`` is used and labelled.

Python boundary: these are per-site time series of a few hundred thousand
sites at most, computed once per forecast with numpy on (time, site)
vectors.  Nothing here touches gridded model fields; the bulk sampling of
wrfout files happens in the Rust site sampler upstream.  This is the small
per-site vector math ``docs/dev/static-rust-port.md`` allows in Python.

Nothing is substituted silently: a product whose required inputs are missing
is refused with the list of what is missing, a product whose site kinds are
absent is skipped with a note, and every approximation is written into the
output's ``notes`` attribute and the printed summary.
"""

from __future__ import annotations

from dataclasses import dataclass
import datetime as _dt
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from woof.energy.contracts import (
    FORECAST_COORDINATES,
    FORECAST_SCHEMA,
    FORECAST_VARIABLES,
    ContractError,
    sha256_file,
)

PRODUCTS = ("dlr", "icing", "wind-power", "pv-power")
PRODUCTS_SCHEMA = "woof-energy.products.v1"
CONDUCTORS_SCHEMA = "woof-energy.conductors.v1"
POWER_CURVES_SCHEMA = "woof-energy.power-curves.v1"

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "energy"
DEFAULT_CONDUCTOR_TABLE = DATA_DIR / "conductors.json"
DEFAULT_POWER_CURVES = DATA_DIR / "power_curves.json"

LINE_KINDS = ("line_sample", "tower")
TURBINE_KINDS = ("turbine", "plant")
PV_KINDS = ("pv",)

#: Products -> site kinds they apply to.  ``icing`` applies to every site.
PRODUCT_KINDS: dict[str, tuple[str, ...]] = {
    "dlr": LINE_KINDS,
    "icing": ("line_sample", "tower", "substation", "turbine", "pv", "plant"),
    "wind-power": TURBINE_KINDS,
    "pv-power": PV_KINDS,
}

#: Required forecast variables (``FORECAST_VARIABLES``) and coordinates
#: (``FORECAST_COORDINATES`` plus the extract-added site coordinates).
#: A tuple inside the list means "any one of these groups".
REQUIRED_VARIABLES: dict[str, list[Any]] = {
    "dlr": ["air_temperature", "wind_speed", "wind_attack_angle"],
    "icing": ["air_temperature", "wind_speed", "cloud_liquid_mixing_ratio",
              "air_density"],
    "wind-power": ["wind_speed",
                   (("air_density",), ("air_pressure", "air_temperature"))],
    "pv-power": ["ghi", "t2", "wind_speed_10m"],
}
REQUIRED_COORDINATES: dict[str, list[str]] = {
    "dlr": ["kind", "lat", "lon", "bearing_deg", "terrain_height"],
    "icing": ["kind"],
    "wind-power": ["kind", "hub_height_m", "capacity_mw"],
    "pv-power": ["kind", "lat", "lon", "capacity_mw"],
}

#: Extra per-site coordinates the extract unit adds to forecast.v1.
SITE_EXTRA_COORDINATES = ("voltage_kv", "hub_height_m", "capacity_mw")

# physical constants
_KELVIN = 273.15
_RHO_WATER = 1000.0
_L_FUSION = 3.34e5          # J kg-1
_L_EVAP = 2.501e6           # J kg-1
_CP_AIR = 1004.0            # J kg-1 K-1
_CW = 4218.0                # J kg-1 K-1, supercooled water
_R_DRY = 287.05             # J kg-1 K-1
_RIME_DENSITY = 500.0       # kg m-3
_GLAZE_DENSITY = 900.0      # kg m-3
_SOLAR_CONSTANT = 1367.0    # W m-2

#: Icing assumptions (documented in the module docstring and attrs).
ICING_MVD_M = 20e-6
ICING_REFERENCE_DIAMETER_M = 0.030
ICING_REFERENCE_HEIGHT_M = 10.0
ICING_DRY_GROWTH_C = -5.0
FREEZING_RAIN_FALL_SPEED = 4.0          # m s-1
FREEZING_RAIN_THRESHOLD = 1e-5          # kg kg-1
ICE_TRACE_KG_M = 1e-3

#: ISO 12494 Table 4 rime icing classes: upper bounds of R1..R9 in kg/m on
#: the reference collector; R10 is anything above 50 kg/m.
ISO12494_RIME_LIMITS_KG_M = (0.5, 0.9, 1.6, 2.8, 5.0, 8.9, 16.0, 28.0, 50.0)

#: DLR solar assumptions.
DLR_ALBEDO = 0.2

#: PV assumptions.
PV_ALBEDO = 0.2
PV_GAMMA_PER_K = -0.004
PV_SYSTEM_LOSSES = 0.14
PV_INVERTER_EFFICIENCY = 0.96
FAIMAN_U0 = 25.0
FAIMAN_U1 = 6.84

#: IEC 61400-12 reference air density.
IEC_REFERENCE_DENSITY = 1.225

#: ``rating_limiting_term`` codes.
LIMIT_UNDEFINED = 0
LIMIT_FORCED_LOW = 1
LIMIT_FORCED_HIGH = 2
LIMIT_NATURAL = 3
LIMIT_NO_HEADROOM = 4
LIMIT_MEANINGS = "undefined forced_convection_low_wind " \
                 "forced_convection_high_wind natural_convection no_headroom"

# IEEE 738-2012 clear-atmosphere total heat flux polynomial (SI, Hc in deg).
_IEEE_CLEAR = (-42.2391, 63.8044, -1.9220, 3.46921e-2, -3.61118e-4,
               1.94318e-6, -4.07608e-9)


class ProductsRefused(ContractError):
    """``woof energy rating`` cannot compute what was asked of it."""


# --------------------------------------------------------------------------
# tables


@dataclass(frozen=True)
class Conductor:
    key: str
    name: str
    diameter_m: float
    r25_ohm_per_m: float
    r75_ohm_per_m: float
    max_temp_c: float
    emissivity: float
    absorptivity: float
    subconductors: int
    height_m: float = 10.0

    def resistance(self, temp_c: float) -> float:
        """Linear interpolation (and extrapolation) of R between 25 and 75."""

        slope = (self.r75_ohm_per_m - self.r25_ohm_per_m) / 50.0
        return self.r25_ohm_per_m + slope * (temp_c - 25.0)


@dataclass
class ConductorTable:
    conductors: dict[str, Conductor]
    auto_rules: list[tuple[float, str]]
    path: Path
    sha256: str

    def get(self, name: str) -> Conductor:
        key = name.strip().lower()
        for conductor in self.conductors.values():
            if key in (conductor.key.lower(), conductor.name.lower()):
                return conductor
        raise ProductsRefused(
            f"unknown conductor {name!r}; the table {self.path.name} has: "
            f"{', '.join(sorted(self.conductors))} (or use auto)")

    def for_voltage(self, voltage_kv: float) -> Conductor | None:
        if not math.isfinite(voltage_kv):
            return None
        for min_kv, key in self.auto_rules:
            if voltage_kv >= min_kv:
                return self.conductors[key]
        return None


def _positive(entry: Mapping[str, Any], name: str, key: str) -> float:
    value = entry.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(value) or value <= 0:
        raise ProductsRefused(
            f"conductor {key!r}: {name} must be a positive number, "
            f"got {value!r}")
    return float(value)


def load_conductor_table(path: str | Path | None = None) -> ConductorTable:
    """Read and validate a ``woof-energy.conductors.v1`` table."""

    path = Path(path) if path is not None else DEFAULT_CONDUCTOR_TABLE
    try:
        document = json.loads(path.read_text())
    except FileNotFoundError as error:
        raise ProductsRefused(f"conductor table {path} does not exist") \
            from error
    except json.JSONDecodeError as error:
        raise ProductsRefused(f"conductor table {path} is not JSON: {error}") \
            from error
    if not isinstance(document, dict) or \
            document.get("schema") != CONDUCTORS_SCHEMA:
        raise ProductsRefused(
            f"conductor table {path} is not a {CONDUCTORS_SCHEMA} document")
    entries = document.get("conductors")
    if not isinstance(entries, dict) or not entries:
        raise ProductsRefused(f"conductor table {path} has no conductors")
    conductors: dict[str, Conductor] = {}
    for key, entry in entries.items():
        if not isinstance(entry, dict):
            raise ProductsRefused(f"conductor {key!r} is not an object")
        sub = entry.get("subconductors", 1)
        if isinstance(sub, bool) or not isinstance(sub, int) or sub < 1:
            raise ProductsRefused(
                f"conductor {key!r}: subconductors must be an integer >= 1")
        emissivity = _positive(entry, "emissivity", key)
        absorptivity = _positive(entry, "absorptivity", key)
        if emissivity > 1 or absorptivity > 1:
            raise ProductsRefused(
                f"conductor {key!r}: emissivity and absorptivity must be "
                "in (0, 1]")
        conductor = Conductor(
            key=str(key), name=str(entry.get("name", key)),
            diameter_m=_positive(entry, "diameter_m", key),
            r25_ohm_per_m=_positive(entry, "r25_ohm_per_m", key),
            r75_ohm_per_m=_positive(entry, "r75_ohm_per_m", key),
            max_temp_c=_positive(entry, "max_temp_c", key),
            emissivity=emissivity, absorptivity=absorptivity,
            subconductors=sub,
            height_m=_positive({"height_m": entry.get("height_m", 10.0)},
                               "height_m", key))
        if conductor.r75_ohm_per_m < conductor.r25_ohm_per_m:
            raise ProductsRefused(
                f"conductor {key!r}: r75_ohm_per_m is below r25_ohm_per_m")
        conductors[str(key)] = conductor
    rules: list[tuple[float, str]] = []
    auto = document.get("auto") or {}
    for rule in auto.get("rules", []) if isinstance(auto, dict) else []:
        try:
            min_kv = float(rule["min_kv"])
            target = str(rule["conductor"])
        except (KeyError, TypeError, ValueError) as error:
            raise ProductsRefused(
                f"conductor table {path}: bad auto rule {rule!r}") from error
        if target not in conductors:
            raise ProductsRefused(
                f"conductor table {path}: auto rule names unknown conductor "
                f"{target!r}")
        rules.append((min_kv, target))
    rules.sort(key=lambda item: -item[0])
    return ConductorTable(conductors=conductors, auto_rules=rules, path=path,
                          sha256=sha256_file(path))


def load_power_curve(path: str | Path | None = None,
                     name: str | None = None) -> dict[str, Any]:
    """Read a normalised power curve from ``power_curves.json``."""

    path = Path(path) if path is not None else DEFAULT_POWER_CURVES
    document = json.loads(path.read_text())
    if document.get("schema") != POWER_CURVES_SCHEMA:
        raise ProductsRefused(
            f"power curves {path} is not a {POWER_CURVES_SCHEMA} document")
    name = name or document["default"]
    try:
        curve = dict(document["curves"][name])
    except KeyError as error:
        raise ProductsRefused(f"power curve {name!r} not in {path}") from error
    speeds = np.asarray(curve["wind_speed_ms"], dtype=float)
    fraction = np.asarray(curve["power_fraction"], dtype=float)
    if speeds.shape != fraction.shape or speeds.size < 2 or \
            np.any(np.diff(speeds) <= 0):
        raise ProductsRefused(f"power curve {name!r} in {path} is malformed")
    curve.update(key=name, speeds=speeds, fraction=fraction,
                 sha256=sha256_file(path))
    return curve


# --------------------------------------------------------------------------
# IEEE 738-2012


def ieee738_solar_angles(lat_deg: Any, day_of_year: Any,
                         solar_hour: Any) -> tuple[np.ndarray, np.ndarray]:
    """Solar altitude ``Hc`` and azimuth ``Zc`` (degrees) per IEEE 738-2012.

    ``solar_hour`` is local apparent solar time in hours (11.0 = 11:00).
    This is the IEEE geometry the Annex B reference is stated in; forecast
    ratings use :func:`solar_position` (NOAA), which takes UTC valid times
    and the equation of time.
    """

    lat = np.radians(np.asarray(lat_deg, dtype=float))
    delta = np.radians(23.46 * np.sin(np.radians(
        (284.0 + np.asarray(day_of_year, dtype=float)) / 365.0 * 360.0)))
    omega_deg = (np.asarray(solar_hour, dtype=float) - 12.0) * 15.0
    omega = np.radians(omega_deg)
    hc = np.degrees(np.arcsin(np.cos(lat) * np.cos(delta) * np.cos(omega)
                              + np.sin(lat) * np.sin(delta)))
    with np.errstate(divide="ignore", invalid="ignore"):
        chi = np.sin(omega) / (np.sin(lat) * np.cos(omega)
                               - np.cos(lat) * np.tan(delta))
    c = np.where(omega_deg < 0, np.where(chi >= 0, 0.0, 180.0),
                 np.where(chi >= 0, 180.0, 360.0))
    zc = c + np.degrees(np.arctan(chi))
    return hc, zc


def ieee738_clear_sky_flux(hc_deg: Any, elevation_m: Any) -> np.ndarray:
    """IEEE 738 clear-atmosphere heat flux ``Q_se`` (W/m2) at elevation."""

    hc = np.asarray(hc_deg, dtype=float)
    qs = np.zeros_like(hc)
    for power, coefficient in enumerate(_IEEE_CLEAR):
        qs = qs + coefficient * hc ** power
    he = np.asarray(elevation_m, dtype=float)
    k_solar = 1.0 + 1.148e-4 * he - 1.108e-8 * he ** 2
    return np.where(hc > 0.0, np.maximum(qs, 0.0) * k_solar, 0.0)


def ieee738_air(film_c: Any, elevation_m: Any
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Air dynamic viscosity, density and thermal conductivity (IEEE 738)."""

    tf = np.asarray(film_c, dtype=float)
    he = np.asarray(elevation_m, dtype=float)
    mu = 1.458e-6 * (tf + 273.0) ** 1.5 / (tf + 383.4)
    rho = (1.293 - 1.525e-4 * he + 6.379e-9 * he ** 2) / (1.0 + 0.00367 * tf)
    k = 2.424e-2 + 7.477e-5 * tf - 4.407e-9 * tf ** 2
    return mu, rho, k


def ieee738_steady_state(*, air_temp_c: Any, wind_speed: Any,
                         attack_angle_deg: Any, elevation_m: Any,
                         solar_heat_w_m: Any, diameter_m: Any,
                         resistance_ohm_m: Any, conductor_temp_c: Any,
                         emissivity: Any) -> dict[str, np.ndarray]:
    """IEEE 738-2012 steady-state ampacity of one (sub)conductor.

    ``attack_angle_deg`` is phi, the angle between the wind and the
    conductor axis (90 = perpendicular).  ``solar_heat_w_m`` is the solar
    gain q_s already multiplied by absorptivity and diameter.  Returns the
    heat terms (W/m), the current (A, 0 where gains exceed losses or the air
    is at or above the conductor temperature) and the limiting-term code.
    """

    ta = np.asarray(air_temp_c, dtype=float)
    ts = np.asarray(conductor_temp_c, dtype=float)
    vw = np.maximum(np.asarray(wind_speed, dtype=float), 0.0)
    phi = np.radians(np.asarray(attack_angle_deg, dtype=float))
    d0 = np.asarray(diameter_m, dtype=float)
    film = 0.5 * (ts + ta)
    mu, rho, k_f = ieee738_air(film, elevation_m)
    dt = ts - ta
    dt_pos = np.maximum(dt, 0.0)
    n_re = d0 * rho * vw / mu
    k_angle = (1.194 - np.cos(phi) + 0.194 * np.cos(2 * phi)
               + 0.368 * np.sin(2 * phi))
    qc1 = k_angle * (1.01 + 1.35 * n_re ** 0.52) * k_f * dt_pos
    qc2 = k_angle * 0.754 * n_re ** 0.6 * k_f * dt_pos
    qcn = 3.645 * np.sqrt(rho) * d0 ** 0.75 * dt_pos ** 1.25
    qc = np.maximum(np.maximum(qc1, qc2), qcn)
    qr = 17.8 * d0 * np.asarray(emissivity, dtype=float) * (
        ((ts + 273.0) / 100.0) ** 4 - ((ta + 273.0) / 100.0) ** 4)
    qs = np.asarray(solar_heat_w_m, dtype=float)
    net = qc + qr - qs
    resistance = np.asarray(resistance_ohm_m, dtype=float)
    with np.errstate(invalid="ignore"):
        amps = np.sqrt(np.maximum(net, 0.0) / resistance)
    term = np.where(qcn >= np.maximum(qc1, qc2), LIMIT_NATURAL,
                    np.where(qc1 >= qc2, LIMIT_FORCED_LOW, LIMIT_FORCED_HIGH))
    no_headroom = (net <= 0.0) | (dt <= 0.0)
    amps = np.where(no_headroom, 0.0, amps)
    term = np.where(no_headroom, LIMIT_NO_HEADROOM, term)
    bad = ~np.isfinite(net) | ~np.isfinite(amps)
    amps = np.where(bad, np.nan, amps)
    term = np.where(bad, LIMIT_UNDEFINED, term).astype(np.int8)
    return {"amps": amps, "q_c": qc, "q_c1": qc1, "q_c2": qc2, "q_cn": qcn,
            "q_r": qr, "q_s": qs, "limiting_term": term}


# --------------------------------------------------------------------------
# solar geometry and irradiance


def solar_position(times: Sequence[_dt.datetime], lat_deg: np.ndarray,
                   lon_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cosine of zenith and azimuth (deg clockwise from north), (time, site).

    NOAA general solar position (Spencer series; about 0.01 rad accuracy),
    with apparent solar time from UTC, longitude and the equation of time.
    """

    doy = np.array([t.timetuple().tm_yday for t in times], dtype=float)
    hours = np.array([t.hour + t.minute / 60.0 + t.second / 3600.0
                      for t in times], dtype=float)
    gamma = 2.0 * np.pi / 365.0 * (doy - 1.0 + (hours - 12.0) / 24.0)
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(gamma)
                       - 0.032077 * np.sin(gamma)
                       - 0.014615 * np.cos(2 * gamma)
                       - 0.040849 * np.sin(2 * gamma))
    decl = (0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
            - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma)
            - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma))
    lat = np.radians(np.asarray(lat_deg, dtype=float))[None, :]
    lon = np.asarray(lon_deg, dtype=float)[None, :]
    true_solar_min = hours[:, None] * 60.0 + eqtime[:, None] + 4.0 * lon
    hour_angle = np.radians(true_solar_min / 4.0 - 180.0)
    decl = decl[:, None]
    cosz = (np.sin(lat) * np.sin(decl)
            + np.cos(lat) * np.cos(decl) * np.cos(hour_angle))
    cosz = np.clip(cosz, -1.0, 1.0)
    azimuth = np.degrees(np.arctan2(
        np.sin(hour_angle),
        np.cos(hour_angle) * np.sin(lat) - np.tan(decl) * np.cos(lat))) + 180.0
    return cosz, np.mod(azimuth, 360.0)


def erbs_decomposition(ghi: np.ndarray, cosz: np.ndarray,
                       day_of_year: np.ndarray
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Split GHI into DNI and DHI with the Erbs et al. (1982) correlation."""

    i0 = _SOLAR_CONSTANT * (1.0 + 0.033 * np.cos(
        2.0 * np.pi * np.asarray(day_of_year, dtype=float) / 365.0))
    up = cosz > 0.065
    with np.errstate(divide="ignore", invalid="ignore"):
        kt = np.where(up, ghi / (i0 * np.where(up, cosz, 1.0)), 0.0)
    kt = np.clip(kt, 0.0, 1.0)
    kd = np.where(kt <= 0.22, 1.0 - 0.09 * kt,
                  np.where(kt <= 0.8,
                           0.9511 - 0.1604 * kt + 4.388 * kt ** 2
                           - 16.638 * kt ** 3 + 12.336 * kt ** 4,
                           0.165))
    dhi = np.where(up, kd * ghi, ghi)
    with np.errstate(divide="ignore", invalid="ignore"):
        dni = np.where(up, (ghi - dhi) / np.where(up, cosz, 1.0), 0.0)
    return np.maximum(dni, 0.0), np.maximum(dhi, 0.0)


# --------------------------------------------------------------------------
# reading forecast.v1


@dataclass
class Forecast:
    path: Path
    times: list[_dt.datetime]
    time_seconds: np.ndarray
    heights: np.ndarray
    coords: dict[str, np.ndarray]
    variables: dict[str, np.ndarray]
    attrs: dict[str, Any]
    time_var: dict[str, Any]

    @property
    def n_site(self) -> int:
        return len(self.coords["site_id"])

    def coord(self, name: str) -> np.ndarray:
        return self.coords[name]


def _floats(var) -> np.ndarray:
    data = var[:]
    if np.ma.isMaskedArray(data):
        return np.ma.filled(data.astype(float), np.nan)
    return np.asarray(data, dtype=float)


def _strings(var) -> np.ndarray:
    import netCDF4

    data = var[:]
    if np.ma.isMaskedArray(data):
        data = data.filled(b"" if data.dtype.kind == "S" else "")
    data = np.asarray(data)
    if data.dtype.kind == "S" and data.ndim == 2:
        data = netCDF4.chartostring(data)
    out = []
    for item in data.ravel():
        if isinstance(item, bytes):
            item = item.decode("utf-8")
        out.append(str(item))
    return np.array(out, dtype=object)


def _is_string(var) -> bool:
    return var.dtype is str or getattr(var.dtype, "kind", "") in ("S", "U", "O")


def _open_forecast(path: Path):
    """Open a forecast.v1 netCDF and check its schema and dimensions."""

    import netCDF4

    if not path.is_file():
        raise ProductsRefused(f"forecast {path} does not exist")
    try:
        dataset = netCDF4.Dataset(str(path), "r")
    except OSError as error:
        raise ProductsRefused(f"forecast {path} is not netCDF: {error}") \
            from error
    try:
        schema = getattr(dataset, "schema", None)
        if schema != FORECAST_SCHEMA:
            raise ProductsRefused(
                f"forecast {path} has schema {schema!r}, expected "
                f"{FORECAST_SCHEMA!r}")
        for dim in ("time", "site", "height"):
            if dim not in dataset.dimensions:
                raise ProductsRefused(
                    f"forecast {path} has no {dim!r} dimension")
            if len(dataset.dimensions[dim]) == 0:
                raise ProductsRefused(
                    f"forecast {path}: the {dim!r} dimension is empty")
        for coord in ("time", "height", "site_id", "kind"):
            if coord not in dataset.variables:
                raise ProductsRefused(
                    f"forecast {path} has no {coord!r} coordinate")
    except BaseException:
        dataset.close()
        raise
    return dataset


@dataclass
class ForecastHeader:
    variables: set[str]
    coords: set[str]
    kind: np.ndarray
    hub_height_m: np.ndarray


def read_forecast_header(path: str | Path) -> ForecastHeader:
    """Names present and the per-site kinds, without reading any data."""

    path = Path(path)
    with _open_forecast(path) as dataset:
        names = set(dataset.variables)
        kind = _strings(dataset.variables["kind"]).astype(str)
        hub = _floats(dataset.variables["hub_height_m"]) \
            if "hub_height_m" in names else np.full(kind.size, np.nan)
    coords = names & (set(FORECAST_COORDINATES) | set(SITE_EXTRA_COORDINATES))
    return ForecastHeader(variables=names & set(FORECAST_VARIABLES),
                          coords=coords, kind=kind, hub_height_m=hub)


def read_forecast(path: str | Path,
                  needed: Sequence[str] | None = None) -> Forecast:
    """Read a ``forecast.v1`` netCDF, checking schema and dimensions.

    ``needed`` limits the data variables read (all of them by default).
    """

    import netCDF4

    path = Path(path)
    with _open_forecast(path) as dataset:
        time_var = dataset.variables["time"]
        units = getattr(time_var, "units", None)
        if not units:
            raise ProductsRefused(f"forecast {path}: time has no units")
        calendar = getattr(time_var, "calendar", "standard")
        raw_time = _floats(time_var)
        try:
            times = netCDF4.num2date(raw_time, units, calendar,
                                     only_use_cftime_datetimes=False,
                                     only_use_python_datetimes=True)
        except (ValueError, TypeError, OverflowError) as error:
            raise ProductsRefused(
                f"forecast {path}: time ({units!r}, calendar {calendar!r}) "
                f"cannot be decoded to UTC datetimes: {error}") from error
        times = [_dt.datetime(t.year, t.month, t.day, t.hour, t.minute,
                              t.second) for t in np.atleast_1d(times)]
        epoch = times[0]
        seconds = np.array([(t - epoch).total_seconds() for t in times])
        if np.any(np.diff(seconds) <= 0):
            raise ProductsRefused(
                f"forecast {path}: time must be strictly increasing")
        heights = _floats(dataset.variables["height"])
        if np.any(np.diff(heights) <= 0):
            raise ProductsRefused(
                f"forecast {path}: height must be strictly increasing")
        coords: dict[str, np.ndarray] = {}
        coordinate_names = [name for name in FORECAST_COORDINATES
                            if name not in ("time", "height")]
        coordinate_names += list(SITE_EXTRA_COORDINATES)
        for name in coordinate_names:
            if name not in dataset.variables:
                continue
            var = dataset.variables[name]
            if var.dimensions[:1] != ("site",):
                raise ProductsRefused(
                    f"forecast {path}: {name} must have the site dimension")
            coords[name] = _strings(var) if _is_string(var) else _floats(var)
        variables: dict[str, np.ndarray] = {}
        wanted = FORECAST_VARIABLES if needed is None else needed
        for name in wanted:
            if name not in dataset.variables or name not in FORECAST_VARIABLES:
                continue
            var = dataset.variables[name]
            dims = FORECAST_VARIABLES[name][0]
            if tuple(var.dimensions) != dims:
                raise ProductsRefused(
                    f"forecast {path}: {name} has dims {var.dimensions}, "
                    f"expected {dims}")
            variables[name] = _floats(var)
        attrs = {key: dataset.getncattr(key) for key in dataset.ncattrs()}
        time_attrs = {key: time_var.getncattr(key)
                      for key in time_var.ncattrs()
                      if key != "_FillValue"}
        time_attrs["_values"] = np.asarray(raw_time)
        time_attrs["_dtype"] = time_var.dtype
    return Forecast(path=path, times=times, time_seconds=seconds,
                    heights=heights, coords=coords, variables=variables,
                    attrs=attrs, time_var=time_attrs)


# --------------------------------------------------------------------------
# vertical interpolation


def interp_height(field: np.ndarray, heights: np.ndarray, target: np.ndarray,
                  *, log: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate a (time, site, height) field to per-site heights.

    Linear in height, or in ln(height) with ``log=True``.  Targets outside
    the forecast range take the nearest height; the second return value is
    the per-site boolean "was outside".
    """

    heights = np.asarray(heights, dtype=float)
    target = np.asarray(target, dtype=float)
    n_site = field.shape[1]
    outside = (target < heights[0]) | (target > heights[-1])
    if heights.size == 1:
        return field[:, :, 0], outside | (target != heights[0])
    z = np.log(heights) if log else heights
    zt = np.log(np.clip(target, heights[0], heights[-1])) if log \
        else np.clip(target, heights[0], heights[-1])
    hi = np.clip(np.searchsorted(z, zt), 1, heights.size - 1)
    lo = hi - 1
    weight = (zt - z[lo]) / (z[hi] - z[lo])
    sites = np.arange(n_site)
    below = field[:, sites, lo]
    above = field[:, sites, hi]
    return below + (above - below) * weight[None, :], outside


# --------------------------------------------------------------------------
# products


@dataclass
class ProductOutput:
    name: str
    dims: tuple[str, ...]
    data: np.ndarray
    attrs: dict[str, Any]


class _Notes(list):
    def add(self, text: str) -> None:
        if text not in self:
            self.append(text)


def _site_mask(forecast: Forecast, kinds: Sequence[str]) -> np.ndarray:
    kind = forecast.coord("kind")
    return np.isin(kind.astype(str), list(kinds))


def _coord_or_nan(forecast: Forecast, name: str) -> np.ndarray:
    if name in forecast.coords:
        return np.asarray(forecast.coords[name], dtype=float)
    return np.full(forecast.n_site, np.nan)


def _wind_mask(kind: np.ndarray, hub_height_m: np.ndarray) -> np.ndarray:
    """Turbines, and plants that carry a hub height (wind farms)."""

    kind = np.asarray(kind).astype(str)
    return (kind == "turbine") | ((kind == "plant")
                                  & np.isfinite(hub_height_m))


def _full(forecast: Forecast, fill: float = np.nan) -> np.ndarray:
    return np.full((len(forecast.times), forecast.n_site), fill)


def _resolve_conductors(forecast: Forecast, mask: np.ndarray,
                        table: ConductorTable, conductor: str,
                        notes: _Notes) -> list[Conductor | None]:
    resolved: list[Conductor | None] = [None] * forecast.n_site
    if conductor.strip().lower() != "auto":
        chosen = table.get(conductor)
        for i in np.flatnonzero(mask):
            resolved[i] = chosen
        return resolved
    voltage = _coord_or_nan(forecast, "voltage_kv")
    missing = 0
    for i in np.flatnonzero(mask):
        resolved[i] = table.for_voltage(float(voltage[i]))
        missing += resolved[i] is None
    if missing:
        floor = min((kv for kv, _ in table.auto_rules), default=float("nan"))
        notes.add(f"dlr: {missing} line site(s) have unknown voltage or are "
                  f"below {floor:g} kV, so --conductor auto assigns no "
                  "conductor and their rating is NaN; pass --conductor to "
                  "rate them")
    return resolved


def _dlr_solar(forecast: Forecast, idx: np.ndarray, conductors: list[Conductor],
               elevation: np.ndarray, notes: _Notes) -> tuple[np.ndarray, str]:
    """Absorbed solar heat q_s (W/m per subconductor), (time, n)."""

    lat = forecast.coord("lat")[idx]
    lon = forecast.coord("lon")[idx]
    bearing = forecast.coord("bearing_deg")[idx]
    diameter = np.array([c.diameter_m for c in conductors])
    absorptivity = np.array([c.absorptivity for c in conductors])
    cosz, azimuth = solar_position(forecast.times, lat, lon)
    altitude = np.degrees(np.arcsin(np.clip(cosz, -1.0, 1.0)))
    up = altitude > 0.0
    cos_eta = np.cos(np.radians(altitude)) * np.cos(
        np.radians(azimuth - bearing[None, :]))
    sin_eta = np.sqrt(np.clip(1.0 - cos_eta ** 2, 0.0, 1.0))
    v = forecast.variables
    if "ghi" in v and "dni" in v and "dhi" in v:
        ghi, dni, dhi = v["ghi"][:, idx], v["dni"][:, idx], v["dhi"][:, idx]
        method = ("forecast ghi/dni/dhi on a horizontal cylinder (CIGRE TB "
                  f"601: beam*sin(eta) + pi/2*dhi + pi/2*albedo*ghi, albedo "
                  f"{DLR_ALBEDO})")
    elif "ghi" in v:
        ghi = v["ghi"][:, idx]
        doy = np.array([t.timetuple().tm_yday for t in forecast.times])
        dni, dhi = erbs_decomposition(ghi, cosz, doy[:, None])
        method = ("forecast ghi split by the Erbs decomposition (no dni/dhi "
                  "in the forecast), then the CIGRE TB 601 cylinder form, "
                  f"albedo {DLR_ALBEDO}")
    else:
        qse = ieee738_clear_sky_flux(altitude, elevation[None, :])
        qs = absorptivity[None, :] * qse * sin_eta * diameter[None, :]
        method = ("IEEE 738-2012 clear-atmosphere solar model (the forecast "
                  "has no irradiance); cloud-free, so ratings are "
                  "conservative under cloud")
        notes.add(f"dlr: solar heating from the {method}")
        return np.where(up, qs, 0.0), method
    beam = np.where(up, np.maximum(dni, 0.0) * sin_eta, 0.0)
    intensity = (beam + 0.5 * np.pi * np.maximum(dhi, 0.0)
                 + 0.5 * np.pi * DLR_ALBEDO * np.maximum(ghi, 0.0))
    notes.add(f"dlr: solar heating from {method}")
    return absorptivity[None, :] * diameter[None, :] * intensity, method


def compute_dlr(forecast: Forecast, table: ConductorTable, conductor: str,
                notes: _Notes) -> list[ProductOutput]:
    mask = _site_mask(forecast, LINE_KINDS)
    resolved = _resolve_conductors(forecast, mask, table, conductor, notes)
    amps = _full(forecast)
    mva = _full(forecast)
    term = np.zeros((len(forecast.times), forecast.n_site), dtype=np.int8)
    names = np.array([c.key if c else "" for c in resolved], dtype=object)
    bearing = forecast.coord("bearing_deg")
    no_bearing = mask & ~np.isfinite(bearing)
    if no_bearing.any():
        notes.add(f"dlr: {int(no_bearing.sum())} line site(s) have no "
                  "bearing_deg, so wind attack angle and beam incidence are "
                  "unknown and their rating is NaN")
    idx = np.array([i for i in np.flatnonzero(mask)
                    if resolved[i] is not None and np.isfinite(bearing[i])],
                   dtype=int)
    solar_method = "none"
    if idx.size:
        conductors = [resolved[i] for i in idx]
        v = forecast.variables
        height = np.array([c.height_m for c in conductors])
        ta, outside = interp_height(v["air_temperature"][:, idx],
                                    forecast.heights, height)
        ws, _ = interp_height(v["wind_speed"][:, idx], forecast.heights,
                              height)
        phi, _ = interp_height(v["wind_attack_angle"][:, idx],
                               forecast.heights, height)
        if outside.any():
            notes.add(f"dlr: conductor height outside the forecast heights "
                      f"{forecast.heights.min():g}-{forecast.heights.max():g}"
                      f" m for {int(outside.sum())} site(s); the nearest "
                      "height was used")
        # With no wind the attack angle is undefined but irrelevant.
        phi = np.where(~np.isfinite(phi) & (ws == 0.0), 90.0, phi)
        elevation = forecast.coord("terrain_height")[idx] + height
        qs, solar_method = _dlr_solar(forecast, idx, conductors, elevation,
                                      notes)
        tmax = np.array([c.max_temp_c for c in conductors])
        result = ieee738_steady_state(
            air_temp_c=ta - _KELVIN, wind_speed=ws, attack_angle_deg=phi,
            elevation_m=elevation[None, :], solar_heat_w_m=qs,
            diameter_m=np.array([c.diameter_m for c in conductors])[None, :],
            resistance_ohm_m=np.array([c.resistance(c.max_temp_c)
                                       for c in conductors])[None, :],
            conductor_temp_c=tmax[None, :],
            emissivity=np.array([c.emissivity for c in conductors])[None, :])
        bundle = np.array([c.subconductors for c in conductors])
        amps[:, idx] = result["amps"] * bundle[None, :]
        term[:, idx] = result["limiting_term"]
        voltage = _coord_or_nan(forecast, "voltage_kv")[idx]
        mva[:, idx] = math.sqrt(3.0) * voltage[None, :] * 1e3 \
            * amps[:, idx] / 1e6
        if not np.all(np.isfinite(voltage)):
            notes.add("dlr: rating_mva is NaN where voltage_kv is unknown")
        notes.add("dlr: rating_mva assumes one circuit per line "
                  "(forecast.v1 does not carry circuit counts)")
    common = {"conductor_table_sha256": table.sha256}
    return [
        ProductOutput("rating_amps", ("time", "site"), amps, {
            "units": "A",
            "long_name": "steady-state thermal rating per phase bundle",
            "method": "IEEE Std 738-2012 steady-state heat balance at the "
                      "conductor's max_temp_c; subconductor rating times "
                      "subconductors",
            "solar_method": solar_method,
            "assumptions": "weather at the conductor height_m (linear in "
                           "height, nearest outside the forecast range); "
                           "no AC/DC skin factor beyond the tabulated R; "
                           "forced convection by max(q_c1, q_c2, q_cn)",
            **common}),
        ProductOutput("rating_mva", ("time", "site"), mva, {
            "units": "MVA",
            "long_name": "thermal rating as apparent power, one circuit",
            "method": "sqrt(3) * voltage_kv * 1e3 * rating_amps / 1e6",
            "assumptions": "one three-phase circuit per line; NaN where "
                           "voltage_kv is unknown",
            **common}),
        ProductOutput("rating_limiting_term", ("time", "site"), term, {
            "long_name": "convective term that sets the rating",
            "flag_values": np.arange(5, dtype=np.int8),
            "flag_meanings": LIMIT_MEANINGS}),
        ProductOutput("conductor", ("site",), names, {
            "long_name": "conductor table entry used for dlr and icing "
                         "(empty where none)"}),
    ]


def finstad_collision_efficiency(diameter_m: np.ndarray, speed: np.ndarray,
                                 air_temp_k: np.ndarray, air_density:
                                 np.ndarray, mvd_m: float = ICING_MVD_M
                                 ) -> np.ndarray:
    """Collision efficiency a1 of Finstad, Lozowski and Gates (1988)."""

    mu = 1.458e-6 * air_temp_k ** 1.5 / (air_temp_k + 110.4)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        k = _RHO_WATER * mvd_m ** 2 * speed / (9.0 * mu * diameter_m)
        re = air_density * speed * mvd_m / mu
        langmuir = np.where(k > 0, re ** 2 / k, 0.0)
        a = 1.066 * k ** -0.00616 * np.exp(-1.103 * k ** -0.688)
        b = 3.641 * k ** -0.498 * np.exp(-1.497 * k ** -0.694)
        c = 0.00637 * np.maximum(langmuir - 100.0, 0.0) ** 0.381
        e = a - 0.028 - c * (b - 0.0454)
    e = np.where((k > 0) & np.isfinite(e), e, 0.0)
    e = np.clip(e, 0.0, 1.0)
    return np.where(np.isfinite(speed) & np.isfinite(air_temp_k), e, np.nan)


def _saturation_vapour_pressure(temp_c: np.ndarray) -> np.ndarray:
    return 611.2 * np.exp(17.67 * temp_c / (temp_c + 243.5))


def accretion_efficiency(flux: np.ndarray, diameter_m: np.ndarray,
                         speed: np.ndarray, air_temp_k: np.ndarray,
                         air_density: np.ndarray,
                         air_pressure: np.ndarray | None) -> np.ndarray:
    """Accretion efficiency a3: 1 for dry growth, else the frozen fraction.

    Simplified Makkonen (2000) heat balance at a 0 degC surface: convective
    cooling, evaporative cooling (saturated air; skipped without pressure)
    and the warming of the collected supercooled water, against the latent
    heat of freezing the whole collected flux.
    """

    tc = air_temp_k - _KELVIN
    mu = 1.458e-6 * air_temp_k ** 1.5 / (air_temp_k + 110.4)
    k_air = 2.424e-2 + 7.477e-5 * tc - 4.407e-9 * tc ** 2
    re = air_density * np.maximum(speed, 0.0) * diameter_m / mu
    nusselt = 0.032 * re ** 0.85
    h = nusselt * k_air / diameter_m
    removal = h * np.pi * diameter_m * (-tc) + flux * _CW * (-tc)
    if air_pressure is not None:
        removal = removal + h * np.pi * diameter_m * 0.62 * _L_EVAP * (
            _saturation_vapour_pressure(0.0)
            - _saturation_vapour_pressure(tc)) / (_CP_AIR * air_pressure)
    with np.errstate(divide="ignore", invalid="ignore"):
        fraction = np.where(flux > 0, removal / (flux * _L_FUSION), 1.0)
    fraction = np.clip(fraction, 0.0, 1.0)
    fraction = np.where(tc <= ICING_DRY_GROWTH_C, 1.0, fraction)
    return np.where(tc < 0.0, fraction, 0.0)


def iso12494_class(mass_kg_m: np.ndarray) -> np.ndarray:
    """ISO 12494 rime class (1-10 for R1-R10) of ice mass per metre; 0 none."""

    mass = np.asarray(mass_kg_m, dtype=float)
    cls = 1 + np.searchsorted(np.array(ISO12494_RIME_LIMITS_KG_M), mass,
                              side="left")
    cls = np.where(mass >= ICE_TRACE_KG_M, cls, 0)
    return np.where(np.isfinite(mass), cls, -1).astype(np.int8)


def accumulate_ice(*, air_temp_k: np.ndarray, speed_normal: np.ndarray,
                   cloud_water: np.ndarray, rain_water: np.ndarray | None,
                   air_density: np.ndarray, air_pressure: np.ndarray | None,
                   diameter_m: np.ndarray, time_seconds: np.ndarray
                   ) -> dict[str, np.ndarray]:
    """Accumulate Makkonen in-cloud and Jones freezing-rain ice over time.

    All fields are (time, site); ``diameter_m`` is the bare collector (site,).
    Rates at successive steps are averaged (trapezoid) over each interval
    and the collector diameter grows with the accreted ice.  No shedding.
    """

    n_time, n_site = air_temp_k.shape
    rime = np.zeros(n_site)
    glaze = np.zeros(n_site)
    mass = np.zeros((n_time, n_site))
    thickness = np.zeros((n_time, n_site))
    rate_out = np.zeros((n_time, n_site))
    freezing = air_temp_k < _KELVIN
    previous: tuple[np.ndarray, np.ndarray] | None = None
    dead = np.zeros(n_site, dtype=bool)
    for step in range(n_time):
        volume = rime / _RIME_DENSITY + glaze / _GLAZE_DENSITY
        diameter = np.sqrt(diameter_m ** 2 + 4.0 * volume / np.pi)
        v = np.maximum(speed_normal[step], 0.0)
        t = air_temp_k[step]
        rho = air_density[step]
        a1 = finstad_collision_efficiency(diameter, v, t, rho)
        w = np.maximum(cloud_water[step], 0.0)
        flux = a1 * w * v * diameter
        pressure = air_pressure[step] if air_pressure is not None else None
        a3 = accretion_efficiency(flux, diameter, v, t, rho, pressure)
        cloud_rate = np.where(freezing[step], flux * a3, 0.0)
        dry = freezing[step] & (a3 >= 1.0)
        rain_rate = np.zeros(n_site)
        if rain_water is not None:
            wr = np.maximum(rain_water[step], 0.0)
            rain_rate = np.where(
                freezing[step],
                diameter * np.hypot(wr * FREEZING_RAIN_FALL_SPEED, wr * v),
                0.0)
        rime_rate = np.where(dry, cloud_rate, 0.0)
        glaze_rate = np.where(dry, 0.0, cloud_rate) + rain_rate
        # Unknown temperature means unknown ice, not "no ice".
        dead |= ~np.isfinite(rime_rate + glaze_rate) | ~np.isfinite(t)
        rime_rate = np.nan_to_num(rime_rate)
        glaze_rate = np.nan_to_num(glaze_rate)
        if previous is not None:
            dt = time_seconds[step] - time_seconds[step - 1]
            rime = rime + 0.5 * (previous[0] + rime_rate) * dt
            glaze = glaze + 0.5 * (previous[1] + glaze_rate) * dt
        previous = (rime_rate, glaze_rate)
        total = rime + glaze
        volume = rime / _RIME_DENSITY + glaze / _GLAZE_DENSITY
        radial = 0.5 * (np.sqrt(diameter_m ** 2 + 4.0 * volume / np.pi)
                        - diameter_m)
        mass[step] = np.where(dead, np.nan, total)
        thickness[step] = np.where(dead, np.nan, radial)
        rate_out[step] = np.where(dead, np.nan, rime_rate + glaze_rate)
    return {"mass": mass, "thickness": thickness, "rate": rate_out}


def compute_icing(forecast: Forecast, table: ConductorTable, conductor: str,
                  notes: _Notes) -> list[ProductOutput]:
    v = forecast.variables
    n_site = forecast.n_site
    line = _site_mask(forecast, LINE_KINDS)
    resolved = _resolve_conductors(forecast, line, table, conductor, _Notes())
    diameter = np.full(n_site, ICING_REFERENCE_DIAMETER_M)
    height = np.full(n_site, ICING_REFERENCE_HEIGHT_M)
    on_conductor = np.zeros(n_site, dtype=bool)
    for i, c in enumerate(resolved):
        if c is not None:
            diameter[i] = c.diameter_m
            height[i] = c.height_m
            on_conductor[i] = True
    if (line & ~on_conductor).any():
        notes.add(f"icing: {int((line & ~on_conductor).sum())} line site(s) "
                  "have no conductor, so ice is accreted on the ISO 12494 "
                  "30 mm reference collector at 10 m")
    if (~line).any():
        notes.add("icing: non-line sites accrete on the ISO 12494 30 mm "
                  "reference collector at 10 m above ground")
    ta, outside = interp_height(v["air_temperature"], forecast.heights, height)
    ws, _ = interp_height(v["wind_speed"], forecast.heights, height)
    rho, _ = interp_height(v["air_density"], forecast.heights, height)
    qc, _ = interp_height(v["cloud_liquid_mixing_ratio"], forecast.heights,
                          height)
    if outside.any():
        notes.add(f"icing: collector height outside the forecast heights for "
                  f"{int(outside.sum())} site(s); the nearest height was used")
    if "wind_attack_angle" in v:
        phi, _ = interp_height(v["wind_attack_angle"], forecast.heights,
                               height)
        normal = np.where(on_conductor[None, :] & np.isfinite(phi),
                          ws * np.sin(np.radians(phi)), ws)
        notes.add("icing: impact speed on conductors is the wind component "
                  "normal to the conductor where the axis is known; the "
                  "reference collector (and conductors of unknown axis) take "
                  "the full speed")
    else:
        normal = ws
        notes.add("icing: no wind_attack_angle in the forecast; impact speed "
                  "is the full wind speed")
    rain = None
    if "rain_mixing_ratio" in v:
        qr, _ = interp_height(v["rain_mixing_ratio"], forecast.heights, height)
        rain = qr * rho
    else:
        notes.add("icing: no rain_mixing_ratio in the forecast; freezing "
                  "rain is not accreted and freezing_rain_flag is absent")
    pressure = None
    if "air_pressure" in v:
        pressure, _ = interp_height(v["air_pressure"], forecast.heights,
                                    height)
    else:
        notes.add("icing: no air_pressure in the forecast; evaporative "
                  "cooling is left out of the wet-growth heat balance "
                  "(less ice frozen near 0 degC)")
    result = accumulate_ice(air_temp_k=ta, speed_normal=normal,
                            cloud_water=qc * rho, rain_water=rain,
                            air_density=rho, air_pressure=pressure,
                            diameter_m=diameter,
                            time_seconds=forecast.time_seconds)
    assumptions = (
        f"Makkonen dM/dt = a1 a2 a3 w v D; a1 Finstad et al. (1988) with "
        f"median volume diameter {ICING_MVD_M * 1e6:g} um; a2 = 1; a3 = 1 at "
        f"or below {ICING_DRY_GROWTH_C:g} degC, else simplified Makkonen heat "
        "balance; freezing rain per Jones (1998) at an assumed "
        f"{FREEZING_RAIN_FALL_SPEED:g} m/s fall speed, all freezing; "
        "no shedding; collector grows with the ice")
    out = [
        ProductOutput("ice_mass", ("time", "site"), result["mass"], {
            "units": "kg m-1", "long_name": "accumulated ice mass per metre",
            "method": "trapezoidal accumulation from the first forecast time",
            "assumptions": assumptions}),
        ProductOutput("ice_thickness", ("time", "site"), result["thickness"], {
            "units": "m", "long_name": "equivalent radial ice thickness",
            "method": f"rime at {_RIME_DENSITY:g} kg m-3, glaze (wet growth "
                      f"and freezing rain) at {_GLAZE_DENSITY:g} kg m-3",
            "assumptions": assumptions}),
        ProductOutput("icing_rate", ("time", "site"), result["rate"], {
            "units": "kg m-1 s-1", "long_name": "ice accretion rate",
            "assumptions": assumptions}),
        ProductOutput("icing_class", ("time", "site"),
                      iso12494_class(result["mass"]), {
            "long_name": "ISO 12494 rime icing class of the accumulated ice",
            "flag_values": np.arange(-1, 11, dtype=np.int8),
            "flag_meanings": "undefined none " + " ".join(
                f"R{i}" for i in range(1, 11)),
            "thresholds_kg_m": np.array(ISO12494_RIME_LIMITS_KG_M),
            "method": "ISO 12494 Table 4 rime class limits applied to the "
                      "accumulated mass per metre on the site's collector; "
                      "glaze is classed on the same mass scale"}),
        ProductOutput("icing_collector_diameter", ("site",), diameter, {
            "units": "m", "long_name": "bare collector diameter used"}),
    ]
    if rain is not None:
        flag = ((ta < _KELVIN) & (qr > FREEZING_RAIN_THRESHOLD)).astype(np.int8)
        flag = np.where(np.isfinite(ta) & np.isfinite(qr), flag, -1)
        out.append(ProductOutput("freezing_rain_flag", ("time", "site"),
                                 flag.astype(np.int8), {
            "long_name": "freezing rain at the collector height",
            "flag_values": np.array([-1, 0, 1], dtype=np.int8),
            "flag_meanings": "undefined no yes",
            "method": f"air temperature below 0 degC and rain mixing ratio "
                      f"above {FREEZING_RAIN_THRESHOLD:g} kg/kg"}))
    return out


def compute_wind_power(forecast: Forecast, curve: Mapping[str, Any],
                       notes: _Notes) -> list[ProductOutput]:
    v = forecast.variables
    kind = forecast.coord("kind").astype(str)
    hub = _coord_or_nan(forecast, "hub_height_m")
    capacity = _coord_or_nan(forecast, "capacity_mw")
    mask = _wind_mask(kind, hub)
    hub_speed = _full(forecast)
    equivalent = _full(forecast)
    power = _full(forecast)
    no_hub = mask & ~np.isfinite(hub)
    if no_hub.any():
        notes.add(f"wind-power: {int(no_hub.sum())} turbine site(s) have no "
                  "hub_height_m; their power is NaN")
    no_cap = mask & np.isfinite(hub) & ~np.isfinite(capacity)
    if no_cap.any():
        notes.add(f"wind-power: {int(no_cap.sum())} turbine site(s) have no "
                  "capacity_mw; hub_wind_speed is computed but power is NaN")
    idx = np.flatnonzero(mask & np.isfinite(hub))
    if idx.size:
        speed, outside = interp_height(v["wind_speed"][:, idx],
                                       forecast.heights, hub[idx], log=True)
        if outside.any():
            notes.add(f"wind-power: hub height outside the forecast heights "
                      f"for {int(outside.sum())} site(s); the nearest height "
                      "was used")
        if "air_density" in v:
            rho, _ = interp_height(v["air_density"][:, idx], forecast.heights,
                                   hub[idx])
        else:
            p, _ = interp_height(v["air_pressure"][:, idx], forecast.heights,
                                 hub[idx])
            t, _ = interp_height(v["air_temperature"][:, idx],
                                 forecast.heights, hub[idx])
            rho = p / (_R_DRY * t)
            notes.add("wind-power: no air_density in the forecast; density "
                      "from air_pressure / (R_d air_temperature), dry air")
        veq = speed * (rho / IEC_REFERENCE_DENSITY) ** (1.0 / 3.0)
        fraction = np.interp(veq, curve["speeds"], curve["fraction"])
        fraction = np.where((veq < curve["cut_in_ms"])
                            | (veq > curve["cut_out_ms"]), 0.0, fraction)
        fraction = np.where(np.isfinite(veq), fraction, np.nan)
        hub_speed[:, idx] = speed
        equivalent[:, idx] = veq
        power[:, idx] = fraction * capacity[idx][None, :]
    curve_note = (f"{curve['key']} (cut-in {curve['cut_in_ms']:g}, rated "
                  f"{curve['rated_ms']:g}, cut-out {curve['cut_out_ms']:g} "
                  "m/s), scaled by capacity_mw")
    return [
        ProductOutput("hub_wind_speed", ("time", "site"), hub_speed, {
            "units": "m s-1", "long_name": "wind speed at hub height",
            "method": "linear in ln(height) between bracketing forecast "
                      "heights; nearest height outside the range"}),
        ProductOutput("hub_equivalent_wind_speed", ("time", "site"),
                      equivalent, {
            "units": "m s-1",
            "long_name": "density-corrected hub wind speed",
            "method": "IEC 61400-12: v (rho / 1.225)^(1/3)"}),
        ProductOutput("wind_power_mw", ("time", "site"), power, {
            "units": "MW", "long_name": "wind power output",
            "method": f"normalised power curve {curve_note}",
            "power_curve_sha256": curve["sha256"],
            "assumptions": "no wake, availability, curtailment or "
                           "turbulence losses"}),
    ]


def compute_pv_power(forecast: Forecast, notes: _Notes
                     ) -> list[ProductOutput]:
    v = forecast.variables
    mask = _site_mask(forecast, PV_KINDS)
    idx = np.flatnonzero(mask)
    poa_out = _full(forecast)
    cell_out = _full(forecast)
    power = _full(forecast)
    lat = forecast.coord("lat")[idx]
    lon = forecast.coord("lon")[idx]
    capacity = _coord_or_nan(forecast, "capacity_mw")[idx]
    if not np.all(np.isfinite(capacity)):
        notes.add(f"pv-power: {int((~np.isfinite(capacity)).sum())} PV "
                  "site(s) have no capacity_mw; their power is NaN")
    cosz_calc, azimuth = solar_position(forecast.times, lat, lon)
    if "cos_solar_zenith" in v:
        cosz = v["cos_solar_zenith"][:, idx]
        notes.add("pv-power: solar zenith from the forecast cos_solar_zenith, "
                  "azimuth from the NOAA solar position")
    else:
        cosz = cosz_calc
        notes.add("pv-power: solar position computed (NOAA algorithm); the "
                  "forecast has no cos_solar_zenith")
    ghi = np.maximum(v["ghi"][:, idx], 0.0)
    if "dni" in v and "dhi" in v:
        dni = np.maximum(v["dni"][:, idx], 0.0)
        dhi = np.maximum(v["dhi"][:, idx], 0.0)
        decomposition = "forecast dni and dhi"
    else:
        doy = np.array([t.timetuple().tm_yday for t in forecast.times])
        dni, dhi = erbs_decomposition(ghi, cosz, doy[:, None])
        decomposition = "Erbs decomposition of forecast ghi (no dni/dhi)"
        notes.add(f"pv-power: beam and diffuse from the {decomposition}")
    tilt = np.radians(0.76 * np.abs(lat) + 3.1)
    surface_azimuth = np.where(lat >= 0.0, 180.0, 0.0)
    sinz = np.sqrt(np.clip(1.0 - cosz ** 2, 0.0, 1.0))
    cos_aoi = (cosz * np.cos(tilt)[None, :] + sinz * np.sin(tilt)[None, :]
               * np.cos(np.radians(azimuth - surface_azimuth[None, :])))
    beam = np.where(cosz > 0.0, dni * np.maximum(cos_aoi, 0.0), 0.0)
    sky = dhi * (1.0 + np.cos(tilt)[None, :]) / 2.0
    ground = ghi * PV_ALBEDO * (1.0 - np.cos(tilt)[None, :]) / 2.0
    poa = beam + sky + ground
    t2 = v["t2"][:, idx] - _KELVIN
    wind = np.maximum(v["wind_speed_10m"][:, idx], 0.0)
    cell = t2 + poa / (FAIMAN_U0 + FAIMAN_U1 * wind)
    dc = poa / 1000.0 * (1.0 + PV_GAMMA_PER_K * (cell - 25.0))
    ac = dc * (1.0 - PV_SYSTEM_LOSSES) * PV_INVERTER_EFFICIENCY
    ac = np.clip(ac, 0.0, 1.0) * capacity[None, :]
    poa_out[:, idx] = poa
    cell_out[:, idx] = cell + _KELVIN
    power[:, idx] = ac
    assumptions = (
        "fixed tilt 0.76|lat|+3.1 deg facing the equator; isotropic sky "
        f"(Liu-Jordan), albedo {PV_ALBEDO:g}; irradiance from "
        f"{decomposition}; no incidence-angle, soiling or shading model "
        "beyond the lumped system losses")
    return [
        ProductOutput("poa_irradiance", ("time", "site"), poa_out, {
            "units": "W m-2", "long_name": "plane-of-array irradiance",
            "method": "beam*cos(aoi) + dhi(1+cos tilt)/2 + "
                      "albedo*ghi(1-cos tilt)/2",
            "assumptions": assumptions}),
        ProductOutput("cell_temperature", ("time", "site"), cell_out, {
            "units": "K", "long_name": "PV cell temperature",
            "method": f"Faiman: t2 + poa/(U0 + U1*wind_speed_10m), U0 "
                      f"{FAIMAN_U0:g} W m-2 K-1, U1 {FAIMAN_U1:g} W m-3 s K-1",
            "assumptions": "10 m wind used as the module-height wind"}),
        ProductOutput("pv_power_mw", ("time", "site"), power, {
            "units": "MW", "long_name": "PV AC power output",
            "method": f"PVWatts-like: poa/1000 (1 {PV_GAMMA_PER_K * 100:+g} "
                      f"%/K (Tcell-25)) x (1-{PV_SYSTEM_LOSSES:g}) x "
                      f"{PV_INVERTER_EFFICIENCY:g} x capacity_mw, clipped at "
                      "capacity_mw",
            "assumptions": assumptions + "; capacity_mw taken as DC "
                           "nameplate"}),
    ]


# --------------------------------------------------------------------------
# checks and orchestration


def _check_inputs(product: str, variables: set[str], coords: set[str]
                  ) -> list[str]:
    missing: list[str] = []
    for need in REQUIRED_VARIABLES[product]:
        if isinstance(need, tuple):
            if not any(all(name in variables for name in group)
                       for group in need):
                missing.append(" or ".join("+".join(group) for group in need))
        elif need not in variables:
            missing.append(need)
    missing += [f"coordinate {name}" for name in REQUIRED_COORDINATES[product]
                if name not in coords]
    return missing


def _write_products(forecast: Forecast, outputs: list[ProductOutput],
                    output: Path, global_attrs: Mapping[str, Any]) -> None:
    import netCDF4

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".partial")
    try:
        _write_dataset(netCDF4, tmp, forecast, outputs, global_attrs)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(output)


def _write_dataset(netCDF4, tmp: Path, forecast: Forecast,
                   outputs: list[ProductOutput],
                   global_attrs: Mapping[str, Any]) -> None:
    with netCDF4.Dataset(str(tmp), "w", format="NETCDF4") as ds:
        ds.createDimension("time", len(forecast.times))
        ds.createDimension("site", forecast.n_site)
        tattrs = dict(forecast.time_var)
        values = tattrs.pop("_values")
        dtype = tattrs.pop("_dtype")
        time = ds.createVariable("time", dtype, ("time",))
        time.setncatts(tattrs)
        time[:] = values
        for name, data in forecast.coords.items():
            dims, units, long_name = FORECAST_COORDINATES.get(
                name, (("site",), None, name))
            if name in SITE_EXTRA_COORDINATES and name not in \
                    FORECAST_COORDINATES:
                units = {"voltage_kv": "kV", "hub_height_m": "m",
                         "capacity_mw": "MW"}[name]
            if data.dtype == object:
                var = ds.createVariable(name, str, ("site",))
                var[:] = np.asarray(data, dtype=object)
            else:
                var = ds.createVariable(name, "f8", ("site",),
                                        fill_value=np.nan)
                var[:] = data
            var.long_name = long_name
            if units:
                var.units = units
        coordinates = " ".join(["time"] + list(forecast.coords))
        for item in outputs:
            data = item.data
            if data.dtype == object:
                var = ds.createVariable(item.name, str, item.dims)
                var[:] = data
            elif data.dtype == np.int8:
                var = ds.createVariable(item.name, "i1", item.dims)
                var[:] = data
            else:
                var = ds.createVariable(item.name, "f8", item.dims,
                                        fill_value=np.nan, zlib=True)
                var[:] = data
            for key, value in item.attrs.items():
                var.setncattr(key, value)
            if item.dims == ("time", "site"):
                var.coordinates = coordinates
        ds.setncatts(dict(global_attrs))


def _compute(forecast_path: Path, *, output: Path, conductor: str = "auto",
             conductor_table: Path | None = None,
             products: Sequence[str] = PRODUCTS
             ) -> tuple[Path, dict[str, Any]]:
    forecast_path = Path(forecast_path)
    output = Path(output)
    products = tuple(dict.fromkeys(products))
    unknown = [p for p in products if p not in PRODUCTS]
    if unknown or not products:
        raise ProductsRefused(f"unknown products {unknown}; choose from "
                              f"{', '.join(PRODUCTS)}")
    if output.resolve() == forecast_path.resolve():
        raise ProductsRefused("the output would overwrite the forecast")
    # Header-level checks first so a wrong file is refused before any read.
    header = read_forecast_header(forecast_path)
    variables, coords = header.variables, header.coords
    notes = _Notes()
    run: list[str] = []
    skipped: dict[str, str] = {}
    refusals: list[str] = []
    for product in products:
        kinds = PRODUCT_KINDS[product]
        if product == "wind-power":
            present = bool(_wind_mask(header.kind, header.hub_height_m).any())
        else:
            present = bool(np.isin(header.kind, list(kinds)).any())
        if not present:
            reason = f"no sites of kind {'/'.join(kinds)}"
            skipped[product] = reason
            notes.add(f"{product}: skipped, {reason}")
            continue
        missing = _check_inputs(product, variables, coords)
        if product == "dlr" and conductor.strip().lower() == "auto" and \
                "voltage_kv" not in coords:
            missing.append("coordinate voltage_kv (needed by --conductor "
                           "auto; or pass --conductor NAME)")
        if missing:
            refusals.append(f"{product} needs {', '.join(missing)}")
        else:
            run.append(product)
    if refusals:
        raise ProductsRefused(
            f"forecast {forecast_path} lacks inputs: " + "; ".join(refusals)
            + ". Re-run woof energy extract with these variables, or drop "
              "the product with --products")
    if not run:
        raise ProductsRefused(
            f"none of {', '.join(products)} applies to the sites in "
            f"{forecast_path}: " + "; ".join(
                f"{k}: {v}" for k, v in skipped.items()))
    table = load_conductor_table(conductor_table)
    if conductor.strip().lower() == "auto":
        if "dlr" in run and not table.auto_rules:
            raise ProductsRefused(
                f"conductor table {table.path} has no auto rules; pass "
                "--conductor NAME")
    elif "dlr" in run or "icing" in run:
        table.get(conductor)
    needed: set[str] = set()
    for product in run:
        for need in REQUIRED_VARIABLES[product]:
            if isinstance(need, tuple):
                for group in need:
                    needed.update(group)
            else:
                needed.add(need)
    optional = {
        "dlr": ("ghi", "dni", "dhi"),
        "icing": ("rain_mixing_ratio", "air_pressure", "wind_attack_angle"),
        "wind-power": (),
        "pv-power": ("dni", "dhi", "cos_solar_zenith"),
    }
    for product in run:
        needed.update(optional[product])
    needed &= variables
    forecast = read_forecast(forecast_path, needed=sorted(needed))
    outputs: list[ProductOutput] = []
    curve = None
    for product in run:
        if product == "dlr":
            outputs += compute_dlr(forecast, table, conductor, notes)
        elif product == "icing":
            outputs += compute_icing(forecast, table, conductor, notes)
        elif product == "wind-power":
            curve = load_power_curve()
            outputs += compute_wind_power(forecast, curve, notes)
        elif product == "pv-power":
            outputs += compute_pv_power(forecast, notes)
    attrs: dict[str, Any] = {
        "schema": PRODUCTS_SCHEMA,
        "title": "woof energy products",
        "forecast": forecast_path.name,
        "forecast_sha256": sha256_file(forecast_path),
        "conductor_table": table.path.name,
        "conductor_table_sha256": table.sha256,
        "conductor": conductor,
        "products": ",".join(run),
        "skipped_products": ",".join(skipped),
        "notes": "\n".join(notes),
        "created": _dt.datetime.now(_dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
    }
    if curve is not None:
        attrs["power_curve"] = curve["key"]
        attrs["power_curve_sha256"] = curve["sha256"]
    for key in ("plan_sha256", "sites_sha256"):
        if key in forecast.attrs:
            attrs[key] = forecast.attrs[key]
    _write_products(forecast, outputs, output, attrs)
    summary = _summary(forecast, outputs, run, skipped, notes, output, attrs)
    return output, summary


def _stat(data: np.ndarray) -> dict[str, float] | None:
    finite = data[np.isfinite(data)]
    if not finite.size:
        return None
    return {"min": round(float(finite.min()), 4),
            "mean": round(float(finite.mean()), 4),
            "max": round(float(finite.max()), 4)}


def _summary(forecast: Forecast, outputs: list[ProductOutput], run: list[str],
             skipped: Mapping[str, str], notes: Sequence[str], output: Path,
             attrs: Mapping[str, Any]) -> dict[str, Any]:
    stats = {}
    for item in outputs:
        if item.dims == ("time", "site") and item.data.dtype.kind == "f":
            stats[item.name] = _stat(item.data)
    kinds, counts = np.unique(forecast.coord("kind").astype(str),
                              return_counts=True)
    return {
        "schema": PRODUCTS_SCHEMA,
        "output": str(output),
        "forecast_sha256": attrs["forecast_sha256"],
        "conductor_table_sha256": attrs["conductor_table_sha256"],
        "times": len(forecast.times),
        "sites": {str(k): int(c) for k, c in zip(kinds, counts)},
        "products": list(run),
        "skipped": dict(skipped),
        "variables": [item.name for item in outputs],
        "stats": stats,
        "notes": list(notes),
    }


def compute_products(forecast_path: Path, *, output: Path,
                     conductor: str = "auto",
                     conductor_table: Path | None = None,
                     products: Sequence[str] = PRODUCTS) -> Path:
    """Compute energy products from a ``forecast.v1`` file into ``output``.

    Raises :class:`ProductsRefused` (a :class:`ContractError`) when the
    forecast, conductor table or requested products cannot be honoured.
    """

    path, _ = _compute(forecast_path, output=output, conductor=conductor,
                       conductor_table=conductor_table, products=products)
    return path


def main(args) -> int:
    try:
        _, summary = _compute(
            Path(args.forecast), output=Path(args.output),
            conductor=args.conductor,
            conductor_table=Path(args.conductor_table)
            if args.conductor_table else None,
            products=tuple(args.products))
    except ContractError as error:
        print(json.dumps({"schema": PRODUCTS_SCHEMA, "refused": str(error)},
                         indent=2))
        return 2
    print(json.dumps(summary, indent=2, default=str))
    return 0


__all__ = [
    "PRODUCTS", "PRODUCTS_SCHEMA", "ProductsRefused", "Conductor",
    "ConductorTable", "load_conductor_table", "load_power_curve",
    "ieee738_solar_angles", "ieee738_clear_sky_flux", "ieee738_air",
    "ieee738_steady_state", "solar_position", "erbs_decomposition",
    "read_forecast", "interp_height", "finstad_collision_efficiency",
    "accretion_efficiency", "accumulate_ice", "iso12494_class",
    "compute_products", "main",
]
