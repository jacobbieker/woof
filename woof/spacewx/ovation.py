"""Auroral energy and number flux: OVATION Prime 2010, and Joule heating.

Layer L3b of the space-weather design (docs/spacewx.md), the part beyond the
oval of :mod:`woof.spacewx.aurora`.

**OVATION Prime 2010** (Newell, Sotirelis & Wing 2009, JGR 114, A09207;
Newell et al. 2010, JGR 115, A03216) is a set of linear regressions of DMSP
SSJ/4-5 precipitation against the Newell solar-wind coupling dPhi/dt, on a
96 MLT x 160 magnetic-latitude grid (0.25 h by 0.5 deg, both hemispheres
from 50 deg poleward), for four seasons and four kinds of aurora: diffuse,
monoenergetic, broadband (wave) electrons, and ions.  This module ports the
model as NOAA distributes it, ovation-prime 2.3.0 (IDL, by P. Newell, JHU/APL,
with updates by R. Redmon and J. Machol, NOAA/NGDC; released as public
domain).  The coefficient files are not shipped: the distribution's zip is
fetched from SourceForge at run time and checked against :data:`OVATION_ZIP`.

What the port reproduces, routine by routine of the IDL:

* ``ap_inter_sol``: dPhi/dt is the Newell coupling of each of the four hours
  up to the time (hour ``h`` is the OMNI2 hour starting at ``h``), averaged
  with weights 0.65^k (k hours back), the current hour's weight further
  multiplied by sqrt(the fraction of the hour elapsed).  An hour whose field or
  speed is missing gets weight 0; fewer than three good hours give no
  coupling.
* ``season_epoch`` and ``prob_estimate``: for each season, flux =
  (b1 + b2 dPhi/dt) x p, where the probability p of that aurora is
  b1p + b2p dPhi/dt clipped to [0, 1], and ions have p = 1.  Where that fit
  is absent (b1p = b2p = 0) the IDL falls back to a table of probabilities by
  dPhi/dt bin, but ``fill_arrays`` reads that table with ``readf`` into a
  subscripted array, which IDL passes by value, so the table stays zero and
  the fallback gives 0.  The distribution's own test outputs show it, those
  from NOAA's later trunk build included, and woof keeps that behaviour (p = 0
  without a fit) and does not read the table: with the table's fallback the
  port misses every electron output by 2-190 % of its peak.  (One trunk
  output, the 12 UT diffuse number flux, has flux in 165 equatorward cells
  where the regression itself is 0 in the run's seasons, so no probability
  gives it; it was made with other tables, and is not used as a reference.)
  Negative fluxes are 0, and the IDL's outlier clamps are kept (electron
  energy flux above 10 erg/cm2/s becomes 0.5, then above 5 becomes 5; see
  :func:`_clamp`).
* ``season_weights``: the four seasons are blended linearly in day of year
  (winter centred on day 354, spring 79, summer 171, fall 263).
* ``draw_je`` with ``n_or_s = 3``: the fit's northern and southern bins at the
  same |MLAT| are averaged (or summed where one is zero), which is the
  combined-hemisphere model the authors recommend.  It is the northern
  hemisphere's aurora; for the southern hemisphere the day of year is moved by
  half a year (the IDL's comment says ``365 - doy``, which leaves the solstices
  in the same season, so woof uses ``doy + 182``, the season of the southern
  hemisphere's own illumination).
* Hemispheric power: the energy flux over the grid's cells (Re = 6371 km),
  leaving out the dayside sub-auroral strip (7-19 MLT, MLAT <= 63 deg) as
  ``draw_je`` does.  The IDL's midnight-wedge interpolation is off in its test
  configuration, and is not ported.

Units: energy flux erg cm-2 s-1 (= mW m-2), number flux cm-2 s-1, power GW.
The model's MLAT is the DMSP AACGM latitude; woof places the grid on the
IGRF-14 centred dipole (:mod:`woof.spacewx.aurora`), as for the oval.

**Joule heating** is Knipp et al. 2004 (Sol. Phys. 224, 495, Table I): the
northern hemisphere's Joule power, a seasonal quadratic in |PC(N)| and |Dst|,
fitted to AMIE.  Knipp et al. double it for the globe; :func:`joule_power`
returns the northern value.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import functools
import hashlib
import io
from pathlib import Path
import zipfile

import numpy as np

from woof.spacewx.assets import PinnedAsset, fetch_pinned
from woof.spacewx.solar_irradiance import naive_utc as _naive_utc

OVATION_ZIP = PinnedAsset(
    name="ovation-prime_2.3.0.zip",
    url="https://downloads.sourceforge.net/project/ovation-prime/distributions/ovation-prime_2.3.0.zip",
    bytes=24358161,
    sha256="93dd4ab2551ac9aa8ba1c0f2bd56ef41179cb31542f6a20fc1b3df31922f3789",
    subdir="ovation",
)
TYPES = ("diff", "mono", "wave", "ions")
SEASONS = ("winter", "spring", "summer", "fall")
NMLT, NMLAT, NDF = 96, 160, 12
#: Hours averaged for dPhi/dt, and the weight per hour back (IDL ap_inter_sol).
COUPLING_HOURS, COUPLING_DECAY = 4, 0.65
EARTH_RADIUS_KM = 6371.0
#: Knipp et al. 2004 Table I: northern Joule power (GW) =
#: a |PC| + b PC^2 + c |Dst| + d Dst^2, by season.
KNIPP_2004 = {
    "annual": (24.89, 3.41, 0.41, 0.0015),
    "winter": (13.36, 5.08, 0.47, 0.0011),
    "summer": (29.27, 8.18, -0.04, 0.0126),
    "equinox": (29.14, 2.54, 0.21, 0.0023),
}
#: The outputs on a grid or on columns.
OUTPUTS = ("eflux_diff", "eflux_mono", "eflux_wave", "eflux_ion",
           "nflux_diff", "nflux_mono", "nflux_wave", "nflux_ion")


class OvationError(ValueError):
    pass


@dataclass(frozen=True)
class Coefficients:
    """The regression tables, indexed [type, season, mlt, mlat]."""
    b1e: np.ndarray        # (4, 4, 96, 160) energy flux intercept
    b2e: np.ndarray        # energy flux slope on dPhi/dt
    b1n: np.ndarray        # number flux intercept
    b2n: np.ndarray        # number flux slope
    b1p: np.ndarray        # (3, 4, 96, 160) probability intercept (electrons only)
    b2p: np.ndarray        # probability slope
    sha256: str
    source: str


def _numbers(text: str) -> np.ndarray:
    try:
        return np.asarray(text.split(), dtype=np.float64)
    except ValueError as exc:
        raise OvationError(f"not an OVATION coefficient file: {exc}") from exc


def _flux_table(text: str, name: str) -> tuple[np.ndarray, np.ndarray]:
    x = _numbers(text)
    rows = x[6:]
    if rows.size != NMLT * NMLAT * 5:
        raise OvationError(f"{name}: {rows.size} numbers after the header, expected {NMLT * NMLAT * 5}")
    rows = rows.reshape(-1, 5)
    i, j = rows[:, 0].astype(np.int64), rows[:, 1].astype(np.int64)
    if i.min() < 0 or i.max() >= NMLT or j.min() < 0 or j.max() >= NMLAT:
        raise OvationError(f"{name}: a bin index lies outside the 96 x 160 grid")
    b1 = np.zeros((NMLT, NMLAT))
    b2 = np.zeros((NMLT, NMLAT))
    b1[i, j] = rows[:, 2]
    b2[i, j] = rows[:, 3]
    return b1, b2


def _prob_table(text: str, name: str) -> tuple[np.ndarray, np.ndarray]:
    """The probability fit (b1p, b2p); the tabulated probabilities after it are
    checked for size but not kept (see the module notes)."""
    x = _numbers(text)
    n = NMLT * NMLAT
    if x.size != 6 + 2 * n + NDF * n:
        raise OvationError(f"{name}: {x.size} numbers, expected {6 + 2 * n + NDF * n}")
    b = x[6:6 + 2 * n].reshape(NMLT, NMLAT, 2)
    return b[..., 0], b[..., 1]


def read_coefficients(source) -> Coefficients:
    """The tables from the distribution's zip, or from a directory holding its
    ``premodel`` files (``{season}_{type}[_n].txt``, ``{season}_prob_b_{type}.txt``)."""
    path = Path(source)
    if path.is_dir():
        premodel = path / "premodel" if (path / "premodel").is_dir() else path

        def read(name):
            return (premodel / name).read_bytes()
        digest = hashlib.sha256()
    else:
        data = path.read_bytes()
        digest = hashlib.sha256(data)
        try:
            archive = zipfile.ZipFile(io.BytesIO(data))
        except zipfile.BadZipFile as exc:
            raise OvationError(f"{path}: not a zip ({exc})") from exc
        members = {Path(m).name: m for m in archive.namelist() if "/premodel/" in f"/{m}"}

        def read(name):
            if name not in members:
                raise OvationError(f"{path}: no premodel/{name}")
            return archive.read(members[name])
    shape = (len(TYPES), len(SEASONS), NMLT, NMLAT)
    b1e, b2e, b1n, b2n = (np.zeros(shape) for _ in range(4))
    b1p, b2p = np.zeros((3, *shape[1:])), np.zeros((3, *shape[1:]))
    for s, season in enumerate(SEASONS):
        for t, kind in enumerate(TYPES):
            for suffix, (one, two) in (("", (b1e, b2e)), ("_n", (b1n, b2n))):
                name = f"{season}_{kind}{suffix}.txt"
                raw = read(name)
                if path.is_dir():
                    digest.update(name.encode() + b"\0" + raw)
                one[t, s], two[t, s] = _flux_table(raw.decode("ascii", "replace"), name)
            if t < 3:
                name = f"{season}_prob_b_{kind}.txt"
                raw = read(name)
                if path.is_dir():
                    digest.update(name.encode() + b"\0" + raw)
                b1p[t, s], b2p[t, s] = _prob_table(raw.decode("ascii", "replace"), name)
    return Coefficients(b1e, b2e, b1n, b2n, b1p, b2p, digest.hexdigest(), str(path))


@functools.lru_cache(maxsize=2)
def _cached(source: str) -> Coefficients:
    return read_coefficients(source)


@functools.lru_cache(maxsize=1)
def _pinned_path() -> str:
    return str(fetch_pinned(OVATION_ZIP))


def load(source=None) -> Coefficients:
    """The pinned distribution (fetched and checked once per process), or a
    local zip or ``premodel`` directory."""
    if source is None or str(source).strip() in ("", "pinned"):
        return _cached(_pinned_path())
    return _cached(str(Path(source).resolve()))


def is_pinned(coef: Coefficients) -> bool:
    """Whether the tables came from the pinned distribution's zip."""
    return coef.sha256 == OVATION_ZIP.sha256


# ------------------------------------------------------------------ drivers

def season_weights(doy: int) -> tuple[float, float, float, float]:
    """(winter, spring, summer, fall) weights for a day of year (IDL season_weights)."""
    doy = int(doy)
    if 79 <= doy < 171:
        summer = 1.0 - (171 - doy) / 92.0
        return 0.0, 1.0 - summer, summer, 0.0
    if 171 <= doy < 263:
        fall = 1.0 - (263 - doy) / 92.0
        return 0.0, 0.0, 1.0 - fall, fall
    if 263 <= doy < 354:
        winter = 1.0 - (354 - doy) / 91.0
        return winter, 0.0, 0.0, 1.0 - winter
    doy0 = doy - 365 if doy >= 354 else doy
    spring = 1.0 - (79 - doy0) / 90.0
    return 1.0 - spring, spring, 0.0, 0.0


def hemisphere_doy(when: datetime, hemisphere: str) -> int:
    """The day of year whose season the hemisphere has (south: half a year on)."""
    doy = _naive_utc(when).timetuple().tm_yday
    if hemisphere == "north":
        return doy
    if hemisphere != "south":
        raise OvationError(f"hemisphere {hemisphere!r}: 'north' or 'south'")
    return (doy + 182 - 1) % 365 + 1


def coupling(wind, when: datetime) -> float:
    """OVATION's dPhi/dt at ``when``: the weighted mean Newell coupling of the
    hour containing ``when`` and the three before it (IDL ap_inter_sol)."""
    from woof.spacewx.solarwind import newell_coupling

    when = _naive_utc(when)
    hour = when.replace(minute=0, second=0, microsecond=0)
    weights, values, good = [], [], 0
    for k in range(COUPLING_HOURS):
        t = np.datetime64(hour - timedelta(hours=k), "h")
        i = int((t - wind.hours[0]).astype(int))
        ok = 0 <= i < wind.hours.size
        if ok:
            v, by, bz = float(wind.speed[i]), float(wind.by[i]), float(wind.bz[i])
            ok = bool(np.isfinite(v) and np.isfinite(by) and np.isfinite(bz) and v > 0
                      and (by != 0.0 or bz != 0.0))
        w = COUPLING_DECAY ** k if ok else 0.0
        values.append(float(newell_coupling(v, by, bz)) if ok else 0.0)
        good += int(ok)
        weights.append(w)
    weights[0] *= np.sqrt((when - hour).total_seconds() / 3600.0)
    if good < 3 or sum(weights) <= 0.0:
        raise OvationError(f"{when}: fewer than 3 of the 4 hours of solar wind up to this time "
                           f"({getattr(wind, 'source', '')})")
    return float(np.dot(weights, values) / sum(weights))


# ------------------------------------------------------------------ the model

def _probability(coef: Coefficients, t: int, s: int, dfdt: float) -> np.ndarray:
    """(96, 160) probability of aurora type ``t`` in season ``s``: the fit
    b1p + b2p dPhi/dt clipped to [0, 1]; 0 where there is no fit (see the module
    notes on prob_estimate's tabulated fallback)."""
    if t == 3:
        return np.ones((NMLT, NMLAT))
    return np.clip(coef.b1p[t, s] + coef.b2p[t, s] * dfdt, 0.0, 1.0)


def _clamp(flux: np.ndarray, t: int, number: bool) -> np.ndarray:
    """The IDL's outlier clamps, applied in its order."""
    if not number and t <= 2:
        flux = np.where(flux > 10.0, 0.5, flux)
        return np.where(flux > 5.0, 5.0, flux)
    if not number:
        flux = np.where(flux > 4.0, 0.25, flux)
        return np.where(flux > 2.0, 2.0, flux)
    if t <= 2:
        flux = np.where(flux > 2.0e10, 0.0, flux)
        return np.where(flux > 2.0e9, 1.0e9, flux)
    flux = np.where(flux > 5.0e8, 0.0, flux)
    return np.where(flux > 1.0e8, 1.0e8, flux)


def fit_grid(coef: Coefficients, dfdt: float, doy: int, t: int, number: bool = False) -> np.ndarray:
    """(96, 160) season-weighted flux of one aurora type on the fit's own grid
    (columns 0-79 south, 80-159 north), before the hemispheres are combined."""
    if not np.isfinite(dfdt) or dfdt < 0.0:
        raise OvationError(f"dPhi/dt {dfdt} is not a finite non-negative coupling")
    b1, b2 = (coef.b1n, coef.b2n) if number else (coef.b1e, coef.b2e)
    out = np.zeros((NMLT, NMLAT))
    for s, w in enumerate(season_weights(doy)):
        if w == 0.0:
            continue
        p = _probability(coef, t, s, dfdt)
        flux = np.where(p > 0.0, np.maximum((dfdt * b2[t, s] + b1[t, s]) * p, 0.0), 0.0)
        out += w * _clamp(flux, t, number)
    return out


def combine_hemispheres(grid: np.ndarray) -> np.ndarray:
    """(96, 80) on |MLAT| = 50 + 0.5 j: the mean of the northern and southern
    bins, or their sum where one is zero (IDL draw_je, n_or_s = 3)."""
    north, south = grid[:, 80:], grid[:, :80]
    return np.where(north * south == 0.0, north + south, 0.5 * (north + south))


def flux_grids(coef: Coefficients, dfdt: float, doy: int) -> dict[str, np.ndarray]:
    """:data:`OUTPUTS` -> (96, 80) combined-hemisphere flux at day of year ``doy``."""
    out = {}
    for t, kind in enumerate(TYPES):
        name = "ion" if kind == "ions" else kind
        out[f"eflux_{name}"] = combine_hemispheres(fit_grid(coef, dfdt, doy, t))
        out[f"nflux_{name}"] = combine_hemispheres(fit_grid(coef, dfdt, doy, t, number=True))
    return out


def hemisphere_grids(coef: Coefficients, dfdt: float, when: datetime) -> dict[str, dict[str, np.ndarray]]:
    """{"north", "south"} -> :func:`flux_grids` in each hemisphere's season."""
    return {h: flux_grids(coef, dfdt, hemisphere_doy(when, h)) for h in ("north", "south")}


def electron_power(grids: dict[str, np.ndarray]) -> float:
    """Hemispheric power of the electrons (diffuse + monoenergetic + broadband), GW."""
    return hemispheric_power(grids["eflux_diff"] + grids["eflux_mono"] + grids["eflux_wave"])


def cell_areas_km2() -> np.ndarray:
    """(96, 80) area of each grid cell, km2, with the IDL's cell edges."""
    mlat1 = 50.001 + 0.5 * np.arange(80)
    theta1 = np.deg2rad(90.0 - mlat1)
    theta2 = np.deg2rad(90.0 - (mlat1 + 0.5))
    band = (2.0 * np.pi / 24.0) * 0.25 * EARTH_RADIUS_KM ** 2 * np.abs(np.cos(theta1) - np.cos(theta2))
    return np.broadcast_to(band, (NMLT, 80))


def hemispheric_power(eflux: np.ndarray) -> float:
    """GW from a (96, 80) energy flux grid (erg cm-2 s-1), without the dayside
    sub-auroral strip (IDL draw_je)."""
    mlt = 0.001 + 0.25 * np.arange(NMLT)[:, None]
    mlat = 50.001 + 0.5 * np.arange(80)[None, :]
    keep = ~((mlt >= 7.0) & (mlt <= 19.0) & (mlat <= 63.0))
    return float(1.0e-6 * np.sum(np.where(keep, cell_areas_km2() * eflux, 0.0)))


def on_columns(grids: dict[str, np.ndarray], mlat, mlt) -> dict[str, np.ndarray]:
    """Each grid sampled at columns of geomagnetic latitude ``mlat`` (deg, either
    sign) and MLT (h): 0 equatorward of 50 deg (IDL mlt_bin / mlat_bin)."""
    mlat = np.abs(np.asarray(mlat, dtype=np.float64))
    mlt = np.mod(np.asarray(mlt, dtype=np.float64), 24.0)
    i = np.clip((mlt * 4.0).astype(np.int64), 0, NMLT - 1)
    inside = (mlat >= 50.0) & (mlat <= 90.0)
    j = np.clip(((np.where(inside, mlat, 50.0) - 50.0) * 2.0).astype(np.int64), 0, 79)
    return {k: np.where(inside, g[i, j], 0.0) for k, g in grids.items()}


#: The fields :func:`energy_columns` gives: electron energy flux (total and
#: by type, erg cm-2 s-1), ion energy flux, electron number flux (cm-2 s-1)
#: and electron mean energy (keV).
COLUMN_FIELDS = ("eflux", "eflux_diff", "eflux_mono", "eflux_wave", "eflux_ion", "nflux", "mean_energy")


def energy_columns(grids: dict[str, dict[str, np.ndarray]], mlat, mlt) -> dict[str, np.ndarray]:
    """:data:`COLUMN_FIELDS` on columns, from :func:`hemisphere_grids`: the
    northern grids where mlat >= 0, else the southern."""
    mlat = np.asarray(mlat, dtype=np.float64)
    north = on_columns(grids["north"], mlat, mlt)
    south = on_columns(grids["south"], mlat, mlt)
    cols = {k: np.where(mlat >= 0.0, north[k], south[k]) for k in north}
    eflux = cols["eflux_diff"] + cols["eflux_mono"] + cols["eflux_wave"]
    nflux = cols["nflux_diff"] + cols["nflux_mono"] + cols["nflux_wave"]
    return {"eflux": eflux, "eflux_diff": cols["eflux_diff"], "eflux_mono": cols["eflux_mono"],
            "eflux_wave": cols["eflux_wave"], "eflux_ion": cols["eflux_ion"], "nflux": nflux,
            "mean_energy": mean_energy_kev(eflux, nflux)}


def mean_energy_kev(eflux, nflux) -> np.ndarray:
    """Mean energy, keV, of a flux with energy flux ``eflux`` (erg cm-2 s-1) and
    number flux ``nflux`` (cm-2 s-1); 0 where there is no number flux."""
    eflux, nflux = np.asarray(eflux, dtype=np.float64), np.asarray(nflux, dtype=np.float64)
    erg_per_kev = 1.602176634e-9
    return np.where(nflux > 0.0, eflux / np.where(nflux > 0.0, nflux, 1.0) / erg_per_kev, 0.0)


# ------------------------------------------------------------------ Joule heating

def knipp_season(when: datetime) -> str:
    """Knipp et al. 2004's northern season: winter 21 Oct - 20 Feb, summer
    21 Apr - 20 Aug, equinox otherwise."""
    when = _naive_utc(when)
    md = (when.month, when.day)
    if md >= (10, 21) or md <= (2, 20):
        return "winter"
    if (4, 21) <= md <= (8, 20):
        return "summer"
    return "equinox"


def joule_power(pc: float, dst: float, when: datetime, season: str | None = None) -> float:
    """Northern-hemisphere Joule power, GW (Knipp et al. 2004, Table I), from the
    northern polar cap index and Dst (nT); ``season="annual"`` uses the all-year fit."""
    pc, dst = abs(float(pc)), abs(float(dst))
    if not (np.isfinite(pc) and np.isfinite(dst)):
        raise OvationError(f"Joule heating needs PC(N) and Dst; got {pc}, {dst}")
    a, b, c, d = KNIPP_2004[season or knipp_season(when)]
    return float(a * pc + b * pc * pc + c * dst + d * dst * dst)


__all__ = ["OVATION_ZIP", "TYPES", "SEASONS", "OUTPUTS", "COLUMN_FIELDS", "KNIPP_2004",
           "Coefficients", "OvationError", "read_coefficients", "load", "is_pinned", "season_weights",
           "hemisphere_doy", "coupling", "fit_grid", "combine_hemispheres", "flux_grids", "hemisphere_grids",
           "electron_power", "cell_areas_km2", "hemispheric_power", "on_columns", "energy_columns",
           "mean_energy_kev", "knipp_season", "joule_power"]
