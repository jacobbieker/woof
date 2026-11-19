"""CRTM atmospheres from woof model columns.

A :class:`Profiles` holds ``n`` columns on ``nlay`` layers ordered top of
atmosphere first, as CRTM wants them: interface pressures ``level_pressure_hpa``
(``n, nlay + 1``), layer pressure, temperature, water vapour (g/kg mass mixing
ratio) and ozone (ppmv), plus the surface CRTM needs for the infrared.

Two doors fill one:

* :func:`profiles_from_wrf_state` takes the WRF-named prognostic fields woof's
  state carries (``T`` perturbation potential temperature, ``P``/``PB``,
  ``QVAPOR``, ``PSFC``, ``P_TOP``, ``TSK``, ``U10``/``V10``, ``LANDMASK``,
  ``XICE``, ``SNOWC``, ``HGT``) at already-selected columns.
* :func:`profiles_from_pressure_levels` takes isobaric analysis fields (a GFS
  or ERA5 background) and puts them on a terrain-following column from the
  surface to the top level, the surface value anchoring the bottom.

woof carries no ozone: unless the caller supplies it, ozone is the US Standard
Atmosphere (1976) profile below, an approximation that moves the 9.6 um band by
kelvins and the water-vapour and window bands by much less.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

#: Dry air over ozone molar mass, for kg/kg to ppmv.
_AIR_OVER_O3 = 28.9644 / 47.9982
#: Dry air over water vapour molar mass.
_EPSILON = 0.621970
_KAPPA = 287.04 / 1004.5

#: Approximate US Standard Atmosphere (1976) ozone, ppmv, against pressure in hPa.
#: Only shape and magnitude matter here (it feeds the 9.6 um band); the table is
#: not a substitute for a model or analysis ozone field.
_O3_STANDARD_HPA = np.array([0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 50.0, 70.0, 100.0,
                             150.0, 200.0, 300.0, 500.0, 700.0, 1100.0])
_O3_STANDARD_PPMV = np.array([0.6, 1.2, 3.5, 5.0, 6.5, 7.8, 7.5, 6.0, 4.6, 3.0, 2.0, 1.0,
                              0.45, 0.25, 0.10, 0.06, 0.04, 0.03])


def standard_ozone_ppmv(pressure_hpa) -> np.ndarray:
    """The approximate US Standard Atmosphere ozone at these pressures (ppmv)."""

    lp = np.log(np.clip(np.asarray(pressure_hpa, dtype=float), _O3_STANDARD_HPA[0], _O3_STANDARD_HPA[-1]))
    return np.interp(lp, np.log(_O3_STANDARD_HPA), _O3_STANDARD_PPMV)


@dataclass
class Profiles:
    """``n`` CRTM columns, top of atmosphere first."""

    level_pressure_hpa: np.ndarray   # (n, nlay + 1)
    pressure_hpa: np.ndarray         # (n, nlay)
    temperature_k: np.ndarray        # (n, nlay)
    water_vapour_gkg: np.ndarray     # (n, nlay), mass mixing ratio
    ozone_ppmv: np.ndarray           # (n, nlay)
    land_fraction: np.ndarray        # (n,)
    water_fraction: np.ndarray
    snow_fraction: np.ndarray
    ice_fraction: np.ndarray
    skin_temperature_k: np.ndarray
    wind_speed_ms: np.ndarray
    wind_direction_deg: np.ndarray
    surface_altitude_m: np.ndarray

    def __post_init__(self) -> None:
        for f in fields(self):
            setattr(self, f.name, np.ascontiguousarray(getattr(self, f.name), dtype=np.float64))
        n, nlay = self.pressure_hpa.shape
        if self.level_pressure_hpa.shape != (n, nlay + 1):
            raise ValueError(f"level_pressure_hpa has shape {self.level_pressure_hpa.shape}, want {(n, nlay + 1)}")
        for name in ("temperature_k", "water_vapour_gkg", "ozone_ppmv"):
            if getattr(self, name).shape != (n, nlay):
                raise ValueError(f"{name} has shape {getattr(self, name).shape}, want {(n, nlay)}")
        for f in fields(self)[5:]:
            if getattr(self, f.name).shape != (n,):
                raise ValueError(f"{f.name} has shape {getattr(self, f.name).shape}, want {(n,)}")
        if np.any(np.diff(self.level_pressure_hpa, axis=1) <= 0):
            raise ValueError("level pressures must increase strictly from the top of the atmosphere down")
        if np.any((self.pressure_hpa <= self.level_pressure_hpa[:, :-1]) |
                  (self.pressure_hpa >= self.level_pressure_hpa[:, 1:])):
            raise ValueError("each layer pressure must lie between its two interfaces")
        cover = self.land_fraction + self.water_fraction + self.snow_fraction + self.ice_fraction
        if np.any(np.abs(cover - 1.0) > 1e-6):
            raise ValueError("surface fractions must sum to 1 in every column")
        if not np.all(np.isfinite(self.temperature_k)) or np.any(self.temperature_k <= 0):
            raise ValueError("temperatures must be finite and positive")

    @property
    def count(self) -> int:
        return self.pressure_hpa.shape[0]

    @property
    def layers(self) -> int:
        return self.pressure_hpa.shape[1]

    def subset(self, index) -> "Profiles":
        return Profiles(**{f.name: getattr(self, f.name)[index] for f in fields(self)})

    def surface_array(self) -> np.ndarray:
        """``(n, 7)`` in the shim's order: land, water, snow, ice, skin T, wind speed, direction."""

        return np.stack([self.land_fraction, self.water_fraction, self.snow_fraction, self.ice_fraction,
                         self.skin_temperature_k, self.wind_speed_ms, self.wind_direction_deg], axis=1)


def _surface_fractions(landmask, xice, snowc):
    land = np.clip(np.asarray(landmask, dtype=float), 0.0, 1.0)
    ice = np.clip(np.asarray(xice, dtype=float), 0.0, 1.0)
    snow = np.clip(np.asarray(snowc, dtype=float), 0.0, 1.0)
    return dict(land_fraction=land * (1 - snow), snow_fraction=land * snow,
                water_fraction=(1 - land) * (1 - ice), ice_fraction=(1 - land) * ice)


def _wind(u10, v10):
    u = np.asarray(u10, dtype=float)
    v = np.asarray(v10, dtype=float)
    # Meteorological convention: the direction the wind blows from, clockwise from north.
    return np.hypot(u, v), np.mod(np.degrees(np.arctan2(-u, -v)), 360.0)


def _interfaces(pressure_hpa, top_hpa, surface_hpa):
    """Interface pressures (n, nlay + 1) for top-first layer pressures."""

    mid = 0.5 * (pressure_hpa[:, 1:] + pressure_hpa[:, :-1])
    return np.concatenate([np.asarray(top_hpa, dtype=float).reshape(-1, 1) * np.ones((len(mid), 1)), mid,
                           np.asarray(surface_hpa, dtype=float).reshape(-1, 1)], axis=1)


def profiles_from_wrf_state(state, *, ozone_ppmv=None, use_theta_m: bool = False) -> Profiles:
    """Columns from WRF-named fields, each ``(n, nz)`` bottom first (3-D) or ``(n,)``.

    ``state`` maps ``T`` (perturbation potential temperature, K), ``P`` and
    ``PB`` (Pa), ``QVAPOR`` (kg/kg), ``PSFC`` (Pa), ``P_TOP`` (Pa, scalar or
    ``(n,)``), ``TSK`` (K), ``U10``/``V10`` (m/s), ``LANDMASK``, ``XICE``,
    ``SNOWC`` (fractions) and ``HGT`` (m).  ``ozone_ppmv`` overrides the
    standard ozone, ``(n, nz)`` bottom first.

    ``use_theta_m`` says ``T + 300`` is WRF's moist potential temperature
    ``theta_m = theta * (1 + (Rv/Rd) * qv)`` (``use_theta_m = 1``, the WRF 4.x
    namelist default) rather than dry theta; then ``theta = (T + 300) / (1 +
    (Rv/Rd) * QVAPOR)`` with ``Rv/Rd = 461.6 / 287.04``.  woof's own state
    carries dry theta by default (``use_theta_m = 0`` in its config), so the
    default here is False; set it when the state came from a moist-theta run.
    Raises ``ValueError`` where ``PSFC`` is not above the lowest mass-level pressure.
    """

    missing = [k for k in ("T", "P", "PB", "QVAPOR", "PSFC", "P_TOP", "TSK", "U10", "V10",
                           "LANDMASK", "XICE", "SNOWC", "HGT") if k not in state]
    if missing:
        raise KeyError(f"WRF state lacks {missing}")
    p = (np.asarray(state["P"], dtype=float) + np.asarray(state["PB"], dtype=float))
    if p.ndim != 2:
        raise ValueError(f"P must be (n, nz), got {p.shape}")
    n = p.shape[0]
    theta = np.asarray(state["T"], dtype=float) + 300.0
    if use_theta_m:
        theta = theta / (1.0 + (461.6 / 287.04) * np.asarray(state["QVAPOR"], dtype=float))
    t = theta * (p / 1.0e5) ** _KAPPA
    q = np.maximum(np.asarray(state["QVAPOR"], dtype=float), 0.0) * 1000.0
    p_hpa = p[:, ::-1] / 100.0
    top = np.broadcast_to(np.asarray(state["P_TOP"], dtype=float).reshape(-1), (n,)) / 100.0
    surface = np.asarray(state["PSFC"], dtype=float).reshape(n) / 100.0
    bad = np.flatnonzero(surface <= p_hpa[:, -1])
    if bad.size:
        raise ValueError(f"PSFC must exceed the lowest mass-level pressure; columns {bad.tolist()} do not")
    o3 = (standard_ozone_ppmv(p_hpa) if ozone_ppmv is None
          else np.asarray(ozone_ppmv, dtype=float)[:, ::-1])
    speed, direction = _wind(state["U10"], state["V10"])
    return Profiles(level_pressure_hpa=_interfaces(p_hpa, top, surface),
                    pressure_hpa=p_hpa, temperature_k=t[:, ::-1], water_vapour_gkg=q[:, ::-1], ozone_ppmv=o3,
                    skin_temperature_k=np.asarray(state["TSK"], dtype=float).reshape(n),
                    wind_speed_ms=speed.reshape(n), wind_direction_deg=direction.reshape(n),
                    surface_altitude_m=np.asarray(state["HGT"], dtype=float).reshape(n),
                    **{k: v.reshape(n) for k, v in
                       _surface_fractions(state["LANDMASK"], state["XICE"], state["SNOWC"]).items()})


def profiles_from_pressure_levels(levels_hpa, temperature_k, specific_humidity, *, surface_pressure_pa,
                                  surface_temperature_k, surface_specific_humidity, skin_temperature_k,
                                  u10, v10, land_fraction, ice_fraction, snow_fraction, surface_altitude_m,
                                  ozone_mixing_ratio=None, layers: int = 60) -> Profiles:
    """Columns from an isobaric analysis on ``layers`` terrain-following layers.

    ``temperature_k``, ``specific_humidity`` (kg/kg) and ``ozone_mixing_ratio``
    (kg/kg, optional) are ``(n, nlev)`` on ``levels_hpa``.  The column runs
    from the top level to the surface pressure on layers equally spaced in
    sigma; between the lowest level above ground and the surface, values
    interpolate (in log pressure) to the 2 m temperature and humidity.
    Isobaric levels below the ground are ignored.
    """

    lev = np.asarray(levels_hpa, dtype=float)
    order = np.argsort(lev)
    lev = lev[order]
    t = np.asarray(temperature_k, dtype=float)[:, order]
    sh = np.asarray(specific_humidity, dtype=float)[:, order]
    o3 = None if ozone_mixing_ratio is None else np.asarray(ozone_mixing_ratio, dtype=float)[:, order]
    n = t.shape[0]
    ps = np.asarray(surface_pressure_pa, dtype=float).reshape(n) / 100.0
    top = lev[0]
    sigma_levels = np.linspace(0.0, 1.0, layers + 1)
    level_p = top + sigma_levels[None, :] * (ps[:, None] - top)
    layer_p = 0.5 * (level_p[:, 1:] + level_p[:, :-1])
    out_t = np.empty((n, layers))
    out_q = np.empty((n, layers))
    out_o3 = np.empty((n, layers))
    t2 = np.asarray(surface_temperature_k, dtype=float).reshape(n)
    q2 = np.asarray(surface_specific_humidity, dtype=float).reshape(n)
    for i in range(n):
        above = lev < ps[i]
        lp = np.log(np.append(lev[above], ps[i]))
        target = np.log(layer_p[i])
        out_t[i] = np.interp(target, lp, np.append(t[i, above], t2[i]))
        q_column = np.interp(target, lp, np.append(sh[i, above], q2[i]))
        out_q[i] = np.maximum(q_column, 0.0) / np.maximum(1.0 - q_column, 1e-6) * 1000.0
        if o3 is not None:
            out_o3[i] = np.interp(target, np.log(lev[above]), o3[i, above]) * 1.0e6 * _AIR_OVER_O3
    if o3 is None:
        out_o3 = standard_ozone_ppmv(layer_p)
    speed, direction = _wind(u10, v10)
    return Profiles(level_pressure_hpa=level_p, pressure_hpa=layer_p, temperature_k=out_t,
                    water_vapour_gkg=out_q, ozone_ppmv=out_o3,
                    skin_temperature_k=np.asarray(skin_temperature_k, dtype=float).reshape(n),
                    wind_speed_ms=speed.reshape(n), wind_direction_deg=direction.reshape(n),
                    surface_altitude_m=np.asarray(surface_altitude_m, dtype=float).reshape(n),
                    **{k: v.reshape(n) for k, v in
                       _surface_fractions(land_fraction, ice_fraction, snow_fraction).items()})
