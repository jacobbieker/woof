"""Plan runs -> ``woof-energy.forecast.v1`` (``woof energy extract``).

The extractor is the last stage before products.  It reads a finished plan,
samples every owning domain's history at the sites it owns, turns the raw
sampler keys into the :data:`~woof.energy.contracts.FORECAST_VARIABLES`
table, merges the domains onto one site axis and one time axis, and writes
netCDF, CSV, Zarr or Icechunk.

Inputs
    * the plan (``plan.v1``); the sites document is ``--sites`` or the
      plan's ``sites.path`` resolved against the plan directory, and a
      recorded ``sites.sha256`` that does not match the file is refused;
    * heights are ``--heights-m`` or the sites document's ``heights_m``;
    * every site in the sites document must be owned by a plan domain
      (``PlanDomain.site_ids``); unowned sites are refused, never dropped.

Sampling
    Each owning domain's files are ``plan.resolve(run_dir) / output_glob``,
    sorted.  WRF topologies go through
    :func:`woof.energy.sample.sample_wrfout`; a ``hex-swath`` mesh goes
    through :func:`woof.energy.sample_hex.sample_mpas` with ``mesh_path``
    taken from ``domain.mesh["mesh_path"]`` (falling back to
    ``domain.mesh["static"]``), a path relative to the plan directory naming
    the culled MPAS static/grid file.  Only keys the history reports through
    ``available_variables`` *and* the requested forecast variables need are
    asked for.

Merging
    Domains are merged onto the intersection of their time axes (noted when
    the axes differ, refused when the intersection is empty).  Precipitation
    rates are differenced on each domain's own axis first, so a rate always
    describes the interval ending at its valid time in the run that made it.

Derivations
    These are per-site vectors (time x owned sites x heights) in numpy.  The
    bulk 3-D work -- reading history, vertical and bilinear interpolation --
    is done by the Rust sampler; what is left here is element-wise algebra
    on arrays the size of the output, which the Python boundary
    (``docs/dev/static-rust-port.md``) allows.

    * winds are earth-relative; ``wind_from_direction`` is
      ``(270 - atan2(v, u) * 180/pi) mod 360``;
    * for a site with bearing ``b`` the conductor unit vector is
      ``(sin b, cos b)``, ``line_normal_wind = |u cos b - v sin b|`` and
      ``wind_attack_angle = asin(min(1, normal / speed))`` in degrees,
      NaN when calm (speed < 1e-6 m/s) or without a bearing;
    * ``T = THETA (PRES / P0) ** (R_d / c_p)`` with WRF's constants from
      :mod:`woof.core.constants`; ``air_density = PRES / (R_d T_v)`` with
      ``T_v = T (1 + 0.608 q)`` and ``q = qv / (1 + qv)``;
    * relative humidity uses vapour pressure ``e = qv p / (eps + qv)``
      (``eps = R_d / R_v``) over Bolton (1980) saturation vapour pressure
      over liquid water, ``e_s = 611.2 exp(17.67 (T - 273.15) / (T -
      29.65))`` Pa -- the same constants WRF's ``SVP1..3`` carry -- and is
      clipped to [0, 100];
    * ``ghi`` is SWDOWN, ``dni`` SWDDNI, ``dhi`` SWDDIF; without SWDDIF but
      with SWDDNI, ``dhi = max(0, GHI - DNI max(cosZ, 0))`` (noted); without
      SWDDNI neither ``dni`` nor ``dhi`` is written -- no decomposition model
      is applied;
    * ``cos_solar_zenith`` is COSZEN when written, else the NOAA general
      solar position algorithm (fractional-year series for the equation of
      time and declination) at each valid time and site (noted);
    * ``precipitation_rate`` is ``d(RAINNC + RAINC) / dt`` in kg m-2 s-1;
      the first step is NaN and accumulation resets (negative differences)
      become NaN (noted).

Every approximation or fallback is written to the variable's ``notes``
attribute and to the dataset's global ``notes``; nothing is substituted
silently.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from woof.core.constants import CP, EP2, P0, RD
from woof.energy.contracts import (
    FORECAST_COORDINATES,
    FORECAST_SCHEMA,
    FORECAST_VARIABLES,
    PROFILE_DIMS,
    ContractError,
    EnergyNotImplemented,
    Plan,
    PlanDomain,
    SiteSet,
    load_plan,
    load_sites,
    sha256_file,
)
from woof.energy.sample import PROFILE_VARS, SURFACE_VARS, SampleUnavailable

FORMATS = ("netcdf", "zarr", "icechunk", "csv")

#: Below this wind speed (m/s) the wind direction and attack angle are NaN.
CALM_SPEED_M_S = 1.0e-6

#: Bolton (1980) saturation vapour pressure constants (Pa, -, K, K).
_SVP_PA = 611.2
_SVP_B = 17.67
_SVP_T0 = 273.15
_SVP_T1 = 29.65
SATURATION_FORMULA = ("Bolton 1980 over liquid water, "
                      "e_s = 611.2 exp(17.67 (T - 273.15) / (T - 29.65)) Pa")
SOLAR_FORMULA = ("NOAA general solar position algorithm (fractional-year "
                 "series for declination and equation of time)")

#: Sampler key alternatives per forecast variable, in order of preference.
#: An empty tuple means "computable without any sampled key".
REQUIREMENTS: dict[str, tuple[tuple[str, ...], ...]] = {
    "u": (("U", "V"),),
    "v": (("U", "V"),),
    "w": (("W",),),
    "wind_speed": (("U", "V"),),
    "wind_from_direction": (("U", "V"),),
    "line_normal_wind": (("U", "V"),),
    "wind_attack_angle": (("U", "V"),),
    "air_temperature": (("THETA", "PRES"),),
    "air_pressure": (("PRES",),),
    "air_density": (("THETA", "PRES", "QVAPOR"),),
    "specific_humidity": (("QVAPOR",),),
    "relative_humidity": (("THETA", "PRES", "QVAPOR"),),
    "cloud_liquid_mixing_ratio": (("QCLOUD",),),
    "rain_mixing_ratio": (("QRAIN",),),
    "ice_mixing_ratio": (("QICE",),),
    "snow_mixing_ratio": (("QSNOW",),),
    "t2": (("T2",),),
    "q2": (("Q2",),),
    "rh2": (("T2", "Q2", "PSFC"),),
    "u10": (("U10",),),
    "v10": (("V10",),),
    "wind_speed_10m": (("U10", "V10"),),
    "psfc": (("PSFC",),),
    "ghi": (("SWDOWN",),),
    "dni": (("SWDDNI",),),
    "dhi": (("SWDDIF",), ("SWDOWN", "SWDDNI", "COSZEN"),
            ("SWDOWN", "SWDDNI")),
    "cos_solar_zenith": (("COSZEN",), ()),
    "precipitation_rate": (("RAINNC", "RAINC"), ("RAINNC",)),
}

#: Fixed method notes written to each variable's ``notes`` attribute.
METHOD_NOTES: dict[str, str] = {
    "wind_from_direction": "(270 - atan2(v, u) * 180/pi) mod 360; NaN when "
                           "calm (speed < 1e-6 m/s)",
    "line_normal_wind": "|u cos b - v sin b| for conductor bearing b; NaN "
                        "where the site has no bearing",
    "wind_attack_angle": "asin(min(1, line_normal_wind / wind_speed)); NaN "
                         "when calm (speed < 1e-6 m/s) or without a bearing",
    "air_temperature": "THETA (PRES / 1e5) ** (R_d / c_p), WRF R_d = 287, "
                       "c_p = 7 R_d / 2",
    "air_density": "PRES / (R_d T (1 + 0.608 q)), q = specific humidity",
    "specific_humidity": "QVAPOR / (1 + QVAPOR)",
    "relative_humidity": f"e = qv p / (R_d/R_v + qv) over {SATURATION_FORMULA}"
                         "; clipped to [0, 100]",
    "rh2": f"from T2, Q2, PSFC: e = q2 psfc / (R_d/R_v + q2) over "
           f"{SATURATION_FORMULA}; clipped to [0, 100]",
    "precipitation_rate": "d(RAINNC + RAINC)/dt over the interval ending at "
                          "each valid time; NaN at the first time",
}


class ExtractRefused(RuntimeError):
    """``woof energy extract`` cannot write a forecast from these inputs."""


@dataclass
class _Notes:
    """Notes keyed by text, remembering which domains raised each one."""

    by_text: dict[str, list[str]] = field(default_factory=dict)

    def add(self, text: str, domain_id: str | None = None) -> None:
        owners = self.by_text.setdefault(text, [])
        if domain_id is not None and domain_id not in owners:
            owners.append(domain_id)

    def render(self, domain_count: int) -> list[str]:
        out = []
        for text, owners in self.by_text.items():
            if owners and domain_count > 1:
                out.append(f"{text} [domains: {', '.join(owners)}]")
            else:
                out.append(text)
        return out


@dataclass
class ForecastData:
    """An in-memory ``forecast.v1`` dataset, ready to write."""

    times: np.ndarray                         # datetime64[s], (T,)
    heights_m: np.ndarray                     # (H,)
    coords: dict[str, np.ndarray]             # name -> (S,) array
    variables: dict[str, np.ndarray]          # name -> (T,S,H) or (T,S)
    variable_notes: dict[str, list[str]]
    notes: list[str]
    attrs: dict[str, str]


# --------------------------------------------------------------------------
# derivations (per-site numpy; see module docstring)


def wind_from_direction(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Meteorological direction the wind blows from, degrees in [0, 360)."""

    direction = np.mod(270.0 - np.degrees(np.arctan2(v, u)), 360.0)
    speed = np.hypot(u, v)
    return np.where(speed < CALM_SPEED_M_S, np.nan, direction)


def line_normal_wind(u: np.ndarray, v: np.ndarray, bearing_deg: np.ndarray
                     ) -> np.ndarray:
    """``|u cos b - v sin b|``; ``bearing_deg`` broadcasts on the site axis
    (shape (S,) against (T, S[, H])).  NaN where the bearing is NaN."""

    bearing = np.radians(_site_axis(bearing_deg, u.ndim))
    return np.abs(u * np.cos(bearing) - v * np.sin(bearing))


def wind_attack_angle(u: np.ndarray, v: np.ndarray, bearing_deg: np.ndarray
                      ) -> np.ndarray:
    """Acute angle between the wind and the conductor axis, 0-90 degrees."""

    speed = np.hypot(u, v)
    normal = line_normal_wind(u, v, bearing_deg)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.minimum(1.0, normal / speed)
        angle = np.degrees(np.arcsin(ratio))
    return np.where(speed < CALM_SPEED_M_S, np.nan, angle)


def air_temperature(theta: np.ndarray, pressure: np.ndarray) -> np.ndarray:
    return theta * (pressure / P0) ** (RD / CP)


def specific_humidity(mixing_ratio: np.ndarray) -> np.ndarray:
    return mixing_ratio / (1.0 + mixing_ratio)


def air_density(pressure: np.ndarray, temperature: np.ndarray,
                mixing_ratio: np.ndarray) -> np.ndarray:
    virtual = temperature * (1.0 + 0.608 * specific_humidity(mixing_ratio))
    return pressure / (RD * virtual)


def saturation_vapour_pressure(temperature: np.ndarray) -> np.ndarray:
    """Bolton (1980) over liquid water, Pa."""

    return _SVP_PA * np.exp(_SVP_B * (temperature - _SVP_T0)
                            / (temperature - _SVP_T1))


def relative_humidity(pressure: np.ndarray, temperature: np.ndarray,
                      mixing_ratio: np.ndarray) -> tuple[np.ndarray, int]:
    """Percent, clipped to [0, 100]; also returns how many values clipped."""

    vapour = mixing_ratio * pressure / (EP2 + mixing_ratio)
    with np.errstate(invalid="ignore"):
        rh = 100.0 * vapour / saturation_vapour_pressure(temperature)
        clipped = int(np.count_nonzero((rh < 0.0) | (rh > 100.0)))
    return np.clip(rh, 0.0, 100.0), clipped


def cos_solar_zenith(times: np.ndarray, lat: np.ndarray, lon: np.ndarray
                     ) -> np.ndarray:
    """NOAA general solar position algorithm; returns (T, S) cosines.

    ``gamma = 2 pi / N (doy - 1 + (hour - 12) / 24)``, the Spencer-series
    declination and equation of time, true solar time from UTC and
    longitude, and ``cos Z = sin(lat) sin(decl) + cos(lat) cos(decl)
    cos(ha)``.  Negative at night (the sun below the horizon).
    """

    times = np.asarray(times, dtype="datetime64[s]")
    years = times.astype("datetime64[Y]")
    doy = (times.astype("datetime64[D]") - years).astype(np.int64) + 1
    year_number = years.astype(np.int64) + 1970
    leap = ((year_number % 4 == 0) & (year_number % 100 != 0)) \
        | (year_number % 400 == 0)
    days_in_year = np.where(leap, 366.0, 365.0)
    seconds = (times - times.astype("datetime64[D]")).astype(np.int64)
    hours = seconds / 3600.0
    gamma = 2.0 * np.pi / days_in_year * (doy - 1 + (hours - 12.0) / 24.0)
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(gamma)
                       - 0.032077 * np.sin(gamma)
                       - 0.014615 * np.cos(2 * gamma)
                       - 0.040849 * np.sin(2 * gamma))
    decl = (0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
            - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma)
            - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma))
    lat_r = np.radians(np.asarray(lat, dtype=np.float64))[None, :]
    lon_d = np.asarray(lon, dtype=np.float64)[None, :]
    true_solar_minutes = hours[:, None] * 60.0 + eqtime[:, None] + 4.0 * lon_d
    hour_angle = np.radians(true_solar_minutes / 4.0 - 180.0)
    cosz = (np.sin(lat_r) * np.sin(decl)[:, None]
            + np.cos(lat_r) * np.cos(decl)[:, None] * np.cos(hour_angle))
    return np.clip(cosz, -1.0, 1.0)


def precipitation_rate(times: np.ndarray, accumulated: np.ndarray
                       ) -> tuple[np.ndarray, int]:
    """``d(accumulated)/dt`` (kg m-2 s-1) along axis 0; returns the rate and
    how many negative differences (bucket resets) were set to NaN."""

    seconds = np.asarray(times, dtype="datetime64[s]").astype(np.int64)
    rate = np.full(accumulated.shape, np.nan, dtype=np.float64)
    if accumulated.shape[0] < 2:
        return rate, 0
    dt = np.diff(seconds).astype(np.float64)
    shape = (-1,) + (1,) * (accumulated.ndim - 1)
    diff = np.diff(accumulated.astype(np.float64), axis=0)
    with np.errstate(invalid="ignore"):
        resets = diff < 0.0
    diff = np.where(resets, np.nan, diff)
    rate[1:] = diff / dt.reshape(shape)
    return rate, int(np.count_nonzero(resets))


def _site_axis(values: np.ndarray, ndim: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values.reshape((1, -1) + (1,) * (ndim - 2))


# --------------------------------------------------------------------------
# variable selection


def _choose(name: str, available: set[str]) -> tuple[str, ...] | None:
    for keys in REQUIREMENTS[name]:
        if set(keys) <= available:
            return keys
    return None


def _why_missing(name: str, available: set[str]) -> str:
    options = [" + ".join(keys) for keys in REQUIREMENTS[name] if keys]
    return (f"{name} needs {' or '.join(options)}; history offers "
            f"{', '.join(sorted(available)) or 'nothing'}")


def select_variables(requested: Sequence[str] | None,
                     available_by_domain: Mapping[str, set[str]]
                     ) -> tuple[list[str], dict[str, dict[str, tuple[str, ...]]],
                                list[str]]:
    """Resolve the forecast variables every domain can supply.

    Returns ``(names, choice[domain][name] -> sampler keys, notes)``.
    Unknown names are refused; explicitly requested names a domain cannot
    supply are refused; defaulted names a domain cannot supply are left out
    with a note.
    """

    explicit = requested is not None
    if explicit:
        unknown = [n for n in requested if n not in FORECAST_VARIABLES]
        if unknown:
            raise ExtractRefused(
                f"unknown forecast variable(s) {', '.join(unknown)}; valid "
                f"names are {', '.join(FORECAST_VARIABLES)}")
        names = list(dict.fromkeys(requested))
    else:
        names = list(FORECAST_VARIABLES)
    choice: dict[str, dict[str, tuple[str, ...]]] = {
        d: {} for d in available_by_domain}
    kept: list[str] = []
    notes: list[str] = []
    refusals: list[str] = []
    for name in names:
        missing = []
        for domain_id, available in available_by_domain.items():
            keys = _choose(name, available)
            if keys is None:
                missing.append(f"domain {domain_id}: "
                               f"{_why_missing(name, available)}")
            else:
                choice[domain_id][name] = keys
        if missing:
            for domain_choice in choice.values():
                domain_choice.pop(name, None)
            if explicit:
                refusals.extend(missing)
            else:
                notes.append(f"{name} not written: " + "; ".join(missing))
        else:
            kept.append(name)
    if refusals:
        raise ExtractRefused("the runs cannot supply the requested "
                             "variable(s): " + "; ".join(refusals))
    if not kept:
        raise ExtractRefused("the runs can supply none of the forecast "
                             "variables: " + "; ".join(notes))
    return kept, choice, notes


def _keys_needed(choice: Mapping[str, tuple[str, ...]], available: set[str]
                 ) -> tuple[list[str], list[str]]:
    keys: set[str] = set()
    for chosen in choice.values():
        keys.update(chosen)
    keys &= available
    profile = [k for k in PROFILE_VARS if k in keys]
    surface = [k for k in SURFACE_VARS if k in keys]
    return profile, surface


# --------------------------------------------------------------------------
# per-domain derivation


def derive_domain(result, choice: Mapping[str, tuple[str, ...]], *,
                  lat: np.ndarray, lon: np.ndarray, bearing_deg: np.ndarray,
                  notes: Callable[[str | None, str], None]
                  ) -> dict[str, np.ndarray]:
    """Forecast variables from one domain's :class:`SampleResult`.

    ``notes(variable_or_None, text)`` records a variable note (or a global
    note when the variable is None).
    """

    profile = {k: np.asarray(v, dtype=np.float64)
               for k, v in result.profile.items()}
    surface = {k: np.asarray(v, dtype=np.float64)
               for k, v in result.surface.items()}

    def need(table, key):
        if key not in table:
            raise ExtractRefused(f"sampler did not return {key} although "
                                 "available_variables listed it")
        return table[key]

    out: dict[str, np.ndarray] = {}
    cache: dict[str, np.ndarray] = {}

    def temperature():
        if "T" not in cache:
            cache["T"] = air_temperature(need(profile, "THETA"),
                                         need(profile, "PRES"))
        return cache["T"]

    def cosz():
        if "cosz" not in cache:
            if "COSZEN" in surface:
                cache["cosz"] = surface["COSZEN"]
            else:
                values = cos_solar_zenith(result.times, lat, lon)
                inside = np.asarray(result.inside, dtype=bool)
                values[:, ~inside] = np.nan
                cache["cosz"] = values
                notes("cos_solar_zenith", "COSZEN not in the history; "
                      f"computed with the {SOLAR_FORMULA} from valid time "
                      "and site latitude/longitude")
        return cache["cosz"]

    for name, keys in choice.items():
        if name in ("u", "v", "wind_speed", "wind_from_direction",
                    "line_normal_wind", "wind_attack_angle"):
            u, v = need(profile, "U"), need(profile, "V")
            if name == "u":
                out[name] = u
            elif name == "v":
                out[name] = v
            elif name == "wind_speed":
                out[name] = np.hypot(u, v)
            elif name == "wind_from_direction":
                out[name] = wind_from_direction(u, v)
            elif name == "line_normal_wind":
                out[name] = line_normal_wind(u, v, bearing_deg)
            else:
                out[name] = wind_attack_angle(u, v, bearing_deg)
        elif name == "w":
            out[name] = need(profile, "W")
        elif name == "air_temperature":
            out[name] = temperature()
        elif name == "air_pressure":
            out[name] = need(profile, "PRES")
        elif name == "air_density":
            out[name] = air_density(need(profile, "PRES"), temperature(),
                                    need(profile, "QVAPOR"))
        elif name == "specific_humidity":
            out[name] = specific_humidity(need(profile, "QVAPOR"))
        elif name == "relative_humidity":
            out[name], clipped = relative_humidity(
                need(profile, "PRES"), temperature(), need(profile, "QVAPOR"))
            if clipped:
                notes("relative_humidity",
                      f"{clipped} value(s) outside [0, 100] clipped")
        elif name in ("cloud_liquid_mixing_ratio", "rain_mixing_ratio",
                      "ice_mixing_ratio", "snow_mixing_ratio"):
            out[name] = need(profile, keys[0])
        elif name in ("t2", "q2", "u10", "v10", "psfc"):
            out[name] = need(surface, keys[0])
        elif name == "rh2":
            out[name], clipped = relative_humidity(
                need(surface, "PSFC"), need(surface, "T2"),
                need(surface, "Q2"))
            if clipped:
                notes("rh2", f"{clipped} value(s) outside [0, 100] clipped")
        elif name == "wind_speed_10m":
            out[name] = np.hypot(need(surface, "U10"), need(surface, "V10"))
        elif name == "ghi":
            out[name] = need(surface, "SWDOWN")
        elif name == "dni":
            out[name] = need(surface, "SWDDNI")
        elif name == "dhi":
            if keys == ("SWDDIF",):
                out[name] = need(surface, "SWDDIF")
            else:
                ghi = need(surface, "SWDOWN")
                dni = need(surface, "SWDDNI")
                mu = np.maximum(cosz(), 0.0)
                out[name] = np.maximum(0.0, ghi - dni * mu)
                source = ("COSZEN" if "COSZEN" in surface else
                          "the computed solar zenith")
                notes("dhi", "SWDDIF not in the history; derived as "
                      f"max(0, SWDOWN - SWDDNI max(cosZ, 0)) using {source}")
        elif name == "cos_solar_zenith":
            out[name] = cosz()
        elif name == "precipitation_rate":
            total = need(surface, "RAINNC")
            if "RAINC" in keys:
                total = total + need(surface, "RAINC")
            else:
                notes("precipitation_rate", "RAINC not in the history; "
                      "rate is grid-scale RAINNC only")
            out[name], resets = precipitation_rate(result.times, total)
            if resets:
                notes("precipitation_rate", f"{resets} negative "
                      "accumulation difference(s) (bucket reset) set to NaN")
        else:  # pragma: no cover - REQUIREMENTS and this table agree
            raise ExtractRefused(f"no derivation for {name}")
    return out


# --------------------------------------------------------------------------
# inputs


def _resolve_sites(plan: Plan, sites_path: Path | None
                   ) -> tuple[Path, SiteSet, str]:
    ref = plan.sites_ref or {}
    if sites_path is None:
        if not ref.get("path"):
            raise ExtractRefused("the plan names no sites document; pass "
                                 "--sites")
        sites_path = Path(ref["path"])
        if not sites_path.is_absolute():
            sites_path = plan.directory / sites_path
    sites_path = Path(sites_path)
    if not sites_path.is_file():
        raise ExtractRefused(f"sites document {sites_path} does not exist")
    digest = sha256_file(sites_path)
    recorded = ref.get("sha256")
    if recorded and recorded != digest:
        raise ExtractRefused(
            f"sites document {sites_path} has sha256 {digest[:12]}..., but "
            f"the plan was built from {recorded[:12]}...; re-plan or pass "
            "the sites document the plan was made from")
    return sites_path, load_sites(sites_path), digest


def _owners(plan: Plan, site_ids: Sequence[str]) -> dict[str, PlanDomain]:
    owner_of: dict[str, PlanDomain] = {}
    for domain in plan.domains:
        for site_id in domain.site_ids:
            owner_of[site_id] = domain
    unowned = [s for s in site_ids if s not in owner_of]
    if unowned:
        shown = ", ".join(unowned[:5])
        more = "" if len(unowned) <= 5 else f" and {len(unowned) - 5} more"
        raise ExtractRefused(
            f"{len(unowned)} site(s) have no owning domain in the plan "
            f"({shown}{more}); re-plan with this sites document")
    return owner_of


def _domain_files(plan: Plan, domain: PlanDomain) -> list[Path]:
    run_dir = plan.resolve(domain.run_dir)
    files = sorted(p for p in run_dir.glob(domain.output_glob) if p.is_file())
    if not files:
        raise ExtractRefused(
            f"domain {domain.domain_id} has no output matching "
            f"{run_dir / domain.output_glob}; run woof energy run first")
    return files


#: ``PlanDomain.mesh`` keys naming the MPAS static/grid file the hex
#: sampler needs, in order of preference.  ``mesh_spec``/``cull_region`` are
#: planning documents, not meshes, and are not passed.
MESH_KEYS = ("mesh_path", "static", "grid")


def _mesh_path(plan: Plan, domain: PlanDomain) -> Path | None:
    mesh = domain.mesh or {}
    for key in MESH_KEYS:
        if mesh.get(key):
            path = Path(mesh[key])
            return path if path.is_absolute() else plan.resolve(str(path))
    return None


@dataclass
class _Sampler:
    available: Callable[[Sequence[Path]], set[str]]
    sample: Callable[..., Any]


def _sampler_for(plan: Plan, domain: PlanDomain) -> _Sampler:
    if domain.topology == "hex-swath":
        from woof.energy import sample_hex

        mesh_path = _mesh_path(plan, domain)
        return _Sampler(
            available=lambda paths: set(sample_hex.available_variables(
                paths, mesh_path=mesh_path)),
            sample=lambda *a, **k: sample_hex.sample_mpas(
                *a, mesh_path=mesh_path, **k))
    from woof.energy import sample

    return _Sampler(available=lambda paths: set(
        sample.available_variables(paths)), sample=sample.sample_wrfout)


# --------------------------------------------------------------------------
# assembly


def build_forecast(plan_path: Path, *, sites_path: Path | None = None,
                   heights_m: Sequence[float] | None = None,
                   variables: Sequence[str] | None = None) -> ForecastData:
    """Sample and merge every owning domain; nothing is written."""

    plan_path = Path(plan_path)
    if not plan_path.is_file():
        raise ExtractRefused(f"plan {plan_path} does not exist")
    plan = load_plan(plan_path)
    sites_path, site_set, sites_digest = _resolve_sites(plan, sites_path)
    if len(site_set) == 0:
        raise ExtractRefused(f"sites document {sites_path} has no sites")
    heights = np.asarray(sorted(set(float(h) for h in heights_m))
                         if heights_m is not None else site_set.heights_m,
                         dtype=np.float64)
    if heights.size == 0 or np.any(heights <= 0.0):
        raise ExtractRefused("heights must be positive metres above ground")

    arrays = site_set.as_arrays()
    site_ids = list(arrays["site_id"])
    owner_of = _owners(plan, site_ids)
    global_notes = _Notes()
    variable_notes: dict[str, _Notes] = {}

    def note(variable: str | None, text: str, domain_id: str | None):
        if variable is None:
            global_notes.add(text, domain_id)
        else:
            variable_notes.setdefault(variable, _Notes()).add(text, domain_id)

    index_of = {s: i for i, s in enumerate(site_ids)}
    stray = sum(1 for d in plan.domains for s in d.site_ids
                if s not in index_of)
    if stray:
        note(None, f"{stray} plan site(s) are not in the sites document "
             "and were not extracted", None)
    owning = [d for d in plan.domains
              if any(s in index_of for s in d.site_ids)]
    if heights_m is not None and tuple(heights) != tuple(site_set.heights_m):
        note(None, "heights from --heights-m override the sites document's "
             f"{list(site_set.heights_m)}", None)

    files: dict[str, list[Path]] = {}
    samplers: dict[str, _Sampler] = {}
    available: dict[str, set[str]] = {}
    for domain in owning:
        files[domain.domain_id] = _domain_files(plan, domain)
        samplers[domain.domain_id] = _sampler_for(plan, domain)
        if domain.topology == "hex-swath" and _mesh_path(plan, domain) is None:
            note(None, f"plan names none of mesh{list(MESH_KEYS)}; the "
                 "MPAS sampler reads the mesh from the history",
                 domain.domain_id)
        available[domain.domain_id] = samplers[domain.domain_id].available(
            files[domain.domain_id])
    names, choice, selection_notes = select_variables(variables, available)
    for text in selection_notes:
        note(None, text, None)
    for name in names:
        if name in METHOD_NOTES:
            note(name, METHOD_NOTES[name], None)

    sampled: list[tuple[PlanDomain, np.ndarray, Any]] = []
    for domain in owning:
        did = domain.domain_id
        members = np.asarray([index_of[s] for s in domain.site_ids
                              if s in index_of], dtype=np.int64)
        profile_keys, surface_keys = _keys_needed(choice[did], available[did])
        result = samplers[did].sample(
            files[did], arrays["lat"][members], arrays["lon"][members],
            list(heights), profile_vars=profile_keys,
            surface_vars=surface_keys)
        returned = np.asarray(result.heights_m, dtype=np.float64)
        if profile_keys and (returned.shape != heights.shape
                             or not np.allclose(returned, heights)):
            raise ExtractRefused(
                f"domain {did}: sampler returned heights {returned.tolist()} "
                f"for requested {heights.tolist()}")
        times = np.asarray(result.times, dtype="datetime64[s]")
        if times.size == 0:
            raise ExtractRefused(f"domain {did}: the history has no times")
        if times.size > 1 and np.any(np.diff(times.astype(np.int64)) <= 0):
            raise ExtractRefused(
                f"domain {did}: history times are not strictly increasing "
                "(overlapping or duplicated output files?)")
        result.times = times
        for text in result.notes:
            note(None, text, did)
        sampled.append((domain, members, result))

    common = sampled[0][2].times
    differ = False
    for _, _, result in sampled[1:]:
        if not np.array_equal(result.times, common):
            differ = True
        common = np.intersect1d(common, result.times)
    if common.size == 0:
        raise ExtractRefused("the owning domains share no valid time; "
                             "their runs do not overlap")
    if differ:
        note(None, f"domain time axes differ; kept their intersection of "
             f"{common.size} time(s), and accumulated fields are "
             "differenced on that common axis", None)

    # Derive on the common axis, so a precipitation rate covers the same
    # interval at every site whatever each run's output frequency was.
    per_domain: list[tuple[PlanDomain, np.ndarray, Any, dict]] = []
    for domain, members, result in sampled:
        did = domain.domain_id
        keep = np.isin(result.times, common)
        result.times = result.times[keep]
        result.profile = {k: np.asarray(v)[keep]
                          for k, v in result.profile.items()}
        result.surface = {k: np.asarray(v)[keep]
                          for k, v in result.surface.items()}
        derived = derive_domain(
            result, choice[did], lat=arrays["lat"][members],
            lon=arrays["lon"][members],
            bearing_deg=arrays["bearing_deg"][members],
            notes=lambda variable, text, did=did: note(variable, text, did))
        per_domain.append((domain, members, result, derived))

    count, ntime, nheight = len(site_ids), common.size, heights.size
    variables_out: dict[str, np.ndarray] = {}
    for name in names:
        dims = FORECAST_VARIABLES[name][0]
        shape = (ntime, count, nheight) if dims == PROFILE_DIMS \
            else (ntime, count)
        variables_out[name] = np.full(shape, np.nan, dtype=np.float64)
    terrain = np.full(count, np.nan)
    dx = np.full(count, np.nan)
    inside = np.zeros(count, dtype=np.int8)
    domain_ids = np.empty(count, dtype=object)
    for domain, members, result, derived in per_domain:
        for name in names:
            values = np.asarray(derived[name], dtype=np.float64)
            variables_out[name][:, members, ...] = values
        terrain[members] = np.asarray(result.terrain_m, dtype=np.float64)
        dx[members] = float(result.dx_m) if result.dx_m else domain.dx_m
        inside[members] = np.asarray(result.inside, dtype=bool).astype(np.int8)
        domain_ids[members] = domain.domain_id
    outside = int(np.count_nonzero(inside == 0))
    if outside:
        note(None, f"{outside} site(s) fall outside their owning domain's "
             "sampled interior (inside = 0, data NaN)", None)

    coords = {
        "site_id": arrays["site_id"], "asset_id": arrays["asset_id"],
        "kind": arrays["kind"], "lat": arrays["lat"], "lon": arrays["lon"],
        "bearing_deg": arrays["bearing_deg"], "terrain_height": terrain,
        "domain_id": domain_ids, "dx_m": dx, "inside": inside,
        "voltage_kv": arrays["voltage_kv"],
        "hub_height_m": arrays["hub_height_m"],
        "capacity_mw": arrays["capacity_mw"],
    }
    domain_count = len(per_domain)
    for name, raised in variable_notes.items():
        if name not in names:
            for text, owners in raised.by_text.items():
                global_notes.by_text.setdefault(
                    f"{name} (used, not written): {text}", list(owners))
    try:
        from woof import __version__ as woof_version
    except ImportError:  # pragma: no cover
        woof_version = None
    attrs = {
        "schema": FORECAST_SCHEMA,
        "plan_sha256": sha256_file(plan_path),
        "sites_sha256": sites_digest,
        "created_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "woof_version": str(woof_version or "unknown"),
        "topology": plan.topology,
        "domains": ",".join(d.domain_id for d, *_ in per_domain),
    }
    return ForecastData(
        times=common, heights_m=heights, coords=coords,
        variables=variables_out,
        variable_notes={n: variable_notes[n].render(domain_count)
                        for n in names if n in variable_notes},
        notes=global_notes.render(domain_count), attrs=attrs)


# --------------------------------------------------------------------------
# writers


def _staging(path: Path, *, store: bool = False) -> Path:
    """Staging path next to ``path``; refuses to replace the wrong kind of
    thing (a directory with a file, a non-Zarr directory with a store)."""

    if store:
        if path.is_file() or (path.is_dir() and any(path.iterdir())
                              and not (path / "zarr.json").is_file()):
            raise ExtractRefused(f"output {path} exists and is not a Zarr "
                                 "store; refusing to replace it")
    elif path.is_dir():
        raise ExtractRefused(f"output {path} is a directory; name a file")
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + ".partial")
    _clear(staging)
    return staging


def _publish(staging: Path, path: Path) -> None:
    """Move ``staging`` into place.  A file is replaced atomically; an
    existing store directory is kept aside until the new one is in place."""

    if path.is_dir() and not path.is_symlink():
        previous = path.with_name(path.name + ".previous")
        _clear(previous)
        os.replace(path, previous)
        try:
            os.replace(staging, path)
        except OSError:
            os.replace(previous, path)
            raise
        shutil.rmtree(previous)
    else:
        os.replace(staging, path)


def _clear(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def _epoch_seconds(times: np.ndarray) -> np.ndarray:
    return np.asarray(times, dtype="datetime64[s]").astype(np.int64)


def _string_coord(name: str) -> bool:
    return name in ("site_id", "asset_id", "kind", "domain_id")


def _site_coordinate_names() -> list[str]:
    return [n for n, (dims, _, _) in FORECAST_COORDINATES.items()
            if dims == ("site",)]


def write_netcdf(data: ForecastData, path: Path) -> Path:
    import netCDF4

    path = Path(path)
    staging = _staging(path)
    site_names = _site_coordinate_names()
    with netCDF4.Dataset(staging, "w", format="NETCDF4") as nc:
        nc.createDimension("time", data.times.size)
        nc.createDimension("site", len(data.coords["site_id"]))
        nc.createDimension("height", data.heights_m.size)
        time = nc.createVariable("time", "i8", ("time",))
        time.units = "seconds since 1970-01-01 00:00:00"
        time.calendar = "proleptic_gregorian"
        time.standard_name = "time"
        time.long_name = FORECAST_COORDINATES["time"][2]
        time[:] = _epoch_seconds(data.times)
        height = nc.createVariable("height", "f8", ("height",))
        height.units = "m"
        height.positive = "up"
        height.long_name = FORECAST_COORDINATES["height"][2]
        height[:] = data.heights_m
        for name in site_names:
            dims, units, long_name = FORECAST_COORDINATES[name]
            values = data.coords[name]
            if _string_coord(name):
                var = nc.createVariable(name, str, dims)
                var[:] = np.asarray([str(v) for v in values], dtype=object)
            elif name == "inside":
                var = nc.createVariable(name, "i1", dims)
                var[:] = np.asarray(values, dtype=np.int8)
            else:
                var = nc.createVariable(name, "f8", dims, zlib=True,
                                        fill_value=np.nan)
                var[:] = np.asarray(values, dtype=np.float64)
            if units:
                var.units = units
            var.long_name = long_name
        coordinates = " ".join(site_names)
        for name, values in data.variables.items():
            dims, units, long_name = FORECAST_VARIABLES[name]
            var = nc.createVariable(name, "f4", dims, zlib=True,
                                    complevel=4, fill_value=np.float32(np.nan))
            var.units = units
            var.long_name = long_name
            var.coordinates = coordinates
            notes = data.variable_notes.get(name)
            if notes:
                var.notes = "; ".join(notes)
            var[:] = values.astype(np.float32)
        for key, value in data.attrs.items():
            nc.setncattr(key, value)
        nc.setncattr("notes", json.dumps(data.notes))
    _publish(staging, path)
    return path


def _csv_column(values: np.ndarray) -> np.ndarray:
    text = np.char.mod("%.7g", np.asarray(values, dtype=np.float64))
    return np.where(np.isfinite(values), text, "")


def write_csv(data: ForecastData, path: Path) -> Path:
    """Long format: one row per (time, site) for surface variables (empty
    ``height_m``) and one per (time, site, height) for profile variables.
    Missing values are empty cells.  Rows are formatted a time step at a
    time with numpy string operations, not cell by cell."""

    path = Path(path)
    staging = _staging(path)
    names = list(data.variables)
    profile = [n for n in names if FORECAST_VARIABLES[n][0] == PROFILE_DIMS]
    stamps = [str(t) + "Z" for t in data.times.astype("datetime64[s]")]
    site_ids = np.asarray([str(s) for s in data.coords["site_id"]],
                          dtype=object)
    count, nheight = site_ids.size, data.heights_m.size
    heights = np.asarray([format(h, "g") for h in data.heights_m],
                         dtype=object)
    columns = 3 + len(names)
    with open(staging, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time", "site_id", "height_m"] + names)
        for t, stamp in enumerate(stamps):
            # Rows for one time: per site, the surface row then its heights.
            per_site = (1 if len(profile) < len(names) else 0) \
                + (nheight if profile else 0)
            block = np.full((count, per_site, columns), "", dtype=object)
            block[:, :, 0] = stamp
            block[:, :, 1] = site_ids[:, None]
            first = per_site - (nheight if profile else 0)
            if profile:
                block[:, first:, 2] = heights[None, :]
            for col, name in enumerate(names, start=3):
                values = data.variables[name][t]
                if name in profile:
                    block[:, first:, col] = _csv_column(values)
                else:
                    block[:, 0, col] = _csv_column(values)
            writer.writerows(block.reshape(-1, columns).tolist())
    _publish(staging, path)
    return path


def to_xarray(data: ForecastData):
    """The forecast as an ``xarray.Dataset`` (xarray imported lazily)."""

    try:
        import xarray as xr
    except ModuleNotFoundError as error:
        raise ExtractRefused(
            "Zarr/Icechunk output needs xarray in the active environment "
            "(install the pixi environment, or write --format netcdf)"
        ) from error
    coords: dict[str, Any] = {
        "time": ("time", data.times.astype("datetime64[ns]"),
                 {"long_name": FORECAST_COORDINATES["time"][2]}),
        "height": ("height", data.heights_m,
                   {"units": "m", "positive": "up",
                    "long_name": FORECAST_COORDINATES["height"][2]}),
    }
    for name in _site_coordinate_names():
        dims, units, long_name = FORECAST_COORDINATES[name]
        values = data.coords[name]
        if _string_coord(name):
            values = np.asarray([str(v) for v in values], dtype=object)
        attrs = {"long_name": long_name}
        if units:
            attrs["units"] = units
        coords[name] = (dims, values, attrs)
    data_vars = {}
    for name, values in data.variables.items():
        dims, units, long_name = FORECAST_VARIABLES[name]
        attrs = {"units": units, "long_name": long_name}
        if data.variable_notes.get(name):
            attrs["notes"] = "; ".join(data.variable_notes[name])
        data_vars[name] = (dims, values.astype(np.float32), attrs)
    attrs = dict(data.attrs)
    attrs["notes"] = json.dumps(data.notes)
    return xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)


def write_zarr(data: ForecastData, path: Path) -> Path:
    try:
        import zarr  # noqa: F401
    except ModuleNotFoundError as error:
        raise ExtractRefused(
            "--format zarr needs zarr in the active environment") from error
    dataset = to_xarray(data)
    path = Path(path)
    staging = _staging(path, store=True)
    dataset.to_zarr(staging, mode="w", consolidated=False, zarr_format=3)
    _publish(staging, path)
    return path


def write_icechunk(data: ForecastData, path: Path) -> tuple[Path, str]:
    """Commit the forecast to an Icechunk repository at ``path`` (branch
    ``main``), the approach of :func:`woof.ml_export.export_zarr_to_icechunk`.
    """

    try:
        import icechunk as ic
    except ModuleNotFoundError as error:
        raise ExtractRefused(
            "--format icechunk needs icechunk in the active environment"
        ) from error
    dataset = to_xarray(data)
    path = Path(path).resolve()
    if path.exists() and not path.is_dir():
        raise ExtractRefused(f"output {path} exists and is not an Icechunk "
                             "repository; refusing to replace it")
    storage = ic.local_filesystem_storage(str(path))
    if path.is_dir() and any(path.iterdir()) \
            and not ic.Repository.exists(storage):
        raise ExtractRefused(f"output {path} is a non-empty directory that "
                             "is not an Icechunk repository")
    path.mkdir(parents=True, exist_ok=True)
    repo = ic.Repository.open_or_create(storage=storage)
    session = repo.writable_session("main")
    dataset.to_zarr(session.store, mode="w", consolidated=False,
                    zarr_format=3)
    snapshot = str(session.commit(
        f"woof energy extract: {FORECAST_SCHEMA} "
        f"({data.times.size} times, {len(data.coords['site_id'])} sites)"))
    return path, snapshot


# --------------------------------------------------------------------------
# entry points


def extract_forecast(plan_path: Path, *, output: Path,
                     sites_path: Path | None = None,
                     heights_m: Sequence[float] | None = None,
                     variables: Sequence[str] | None = None,
                     fmt: str = "netcdf") -> Path:
    """Sample the plan's runs at the sites and write ``forecast.v1``."""

    path, _ = _extract(plan_path, output=output, sites_path=sites_path,
                       heights_m=heights_m, variables=variables, fmt=fmt)
    return path


def _extract(plan_path, *, output, sites_path, heights_m, variables, fmt
             ) -> tuple[Path, dict[str, Any]]:
    if fmt not in FORMATS:
        raise ExtractRefused(f"format {fmt!r} is not one of {FORMATS}")
    data = build_forecast(Path(plan_path),
                          sites_path=None if sites_path is None
                          else Path(sites_path),
                          heights_m=heights_m, variables=variables)
    output = Path(output)
    extra: dict[str, Any] = {}
    try:
        if fmt == "netcdf":
            written = write_netcdf(data, output)
        elif fmt == "csv":
            written = write_csv(data, output)
        elif fmt == "zarr":
            written = write_zarr(data, output)
        else:
            written, extra["snapshot"] = write_icechunk(data, output)
    except BaseException:
        _clear(output.with_name(output.name + ".partial"))
        raise
    times = data.times.astype("datetime64[s]")
    summary = {
        "schema": FORECAST_SCHEMA,
        "output": str(written),
        "format": fmt,
        "sites": len(data.coords["site_id"]),
        "sites_inside": int(np.count_nonzero(data.coords["inside"])),
        "times": int(times.size),
        "first_time": str(times[0]) + "Z",
        "last_time": str(times[-1]) + "Z",
        "heights_m": data.heights_m.tolist(),
        "domains": data.attrs["domains"].split(","),
        "variables": list(data.variables),
        "variable_notes": data.variable_notes,
        "notes": data.notes,
        **extra,
    }
    return written, summary


def main(args) -> int:
    try:
        _, summary = _extract(
            Path(args.plan), output=Path(args.output),
            sites_path=None if args.sites is None else Path(args.sites),
            heights_m=args.heights_m, variables=args.vars, fmt=args.format)
    except (ExtractRefused, ContractError, EnergyNotImplemented,
            SampleUnavailable) as error:
        print(f"woof energy extract: refused: {error}", file=sys.stderr)
        return 2
    except OSError as error:
        print(f"woof energy extract: failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, default=str))
    return 0


__all__ = [
    "FORMATS", "REQUIREMENTS", "ExtractRefused", "ForecastData",
    "build_forecast", "extract_forecast", "main", "select_variables",
    "derive_domain", "wind_from_direction", "line_normal_wind",
    "wind_attack_angle", "air_temperature", "air_density",
    "specific_humidity", "relative_humidity", "saturation_vapour_pressure",
    "cos_solar_zenith", "precipitation_rate", "write_netcdf", "write_csv",
    "write_zarr", "write_icechunk", "to_xarray",
]
