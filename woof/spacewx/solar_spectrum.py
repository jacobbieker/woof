"""Solar spectral irradiance (SSI) variability for RRTMG's shortwave bands.

``solar_tsi_opt`` scales every shortwave output by TSI(t) / reference: the
spectrum keeps RRTMG's fixed shape (Kurucz, in ``sfluxref``).  Over the solar
cycle the ultraviolet varies far more than the visible (200-260 nm by several
per cent, 400-700 nm by about 0.1 %), so the shape changes too.
``solar_ssi_opt = 1`` multiplies each of RRTMG SW's 14 bands (16-29) by

    f_b(t) = [S_b(t) / TSI(t)] / [S_b,ref / TSI_ref]

renormalised so that sum_b w_b f_b = sum_b w_b with w_b = S_b,ref / TSI_ref:
the band shares move, and the total stays the TSI factor's.  S_b is the NOAA
CDR NRLSSI2 daily SSI integrated over the band; the reference is the mean over
solar cycle 24 (2009-2019), shipped as :data:`REF_BAND_FRACTION` and refitted
by ``tools/spacewx_ssi_reference.py``.  RRTMG's own band weights (Kurucz)
are within 1.2 % of NRLSSI2's except 263-345 nm (+7 %), so the total of RRTMG's
source moves by sum_b (wK_b - w_b) (f_b - 1), about 1e-5 of the TSI.

The factor reaches the scheme as ``band_factor`` on
``rrtmg_sw_batched_device`` (it multiplies ``adjflux``), for RRTMG
(``ra_sw_physics = 4``) and RRTMG-K (14).  Daily means are placed at noon UTC
and interpolated linearly; a time outside the series is refused.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import os
from pathlib import Path

import numpy as np

from woof.spacewx.solar_irradiance import (_EPOCH_1610, SolarForcingError, cdr_day_centres, fetch_cdr_years,
                                             naive_utc, noon_years)

#: RRTMG SW bands 16..29 (rrsw_wvn wavenum1/wavenum2), cm-1.
BANDS_CM1 = ((2600.0, 3250.0), (3250.0, 4000.0), (4000.0, 4650.0), (4650.0, 5150.0),
             (5150.0, 6150.0), (6150.0, 7700.0), (7700.0, 8050.0), (8050.0, 12850.0),
             (12850.0, 16000.0), (16000.0, 22650.0), (22650.0, 29000.0), (29000.0, 38000.0),
             (38000.0, 50000.0), (820.0, 2600.0))
NBANDS = len(BANDS_CM1)
#: The bands' edges in nm (low, high).
BANDS_NM = tuple((1.0e7 / hi, 1.0e7 / lo) for lo, hi in BANDS_CM1)
#: S_b,ref / TSI_ref: the NRLSSI2 v03 band shares averaged over 2009-01-01 to
#: 2019-12-31 (tools/spacewx_ssi_reference.py; NOAA CDR, open data).
REF_BAND_FRACTION = (0.00884400, 0.01482416, 0.01726081, 0.01624697, 0.04018645, 0.07476922, 0.01767045,
                     0.25270815, 0.15969940, 0.25604600, 0.09551557, 0.03421286, 0.00219724, 0.00940440)
#: NOAA CDR daily SSI (NRLSSI2 v03, standard resolution), one netCDF per year.
NRLSSI2_SSI_INDEX_URL = ("https://www.ncei.noaa.gov/data/solar-spectral-irradiance/access/"
                         "standard-resolution/daily/")
SOURCES = ("nrlssi2",)


def band_irradiance(wavelength_nm, width_nm, ssi) -> np.ndarray:
    """(..., 14) W m-2 in RRTMG's bands from SSI (..., n) in W m-2 nm-1 on bins
    centred at ``wavelength_nm`` of width ``width_nm`` (partial bins pro rata)."""
    return np.asarray(ssi, np.float64) @ _band_weights(wavelength_nm, width_nm).T


@dataclass(frozen=True)
class SsiSeries:
    """Daily band irradiances (W m-2) and TSI at noon UTC of each day, ascending."""

    days: np.ndarray         # float64 days since 1610-01-01 00 UTC
    band: np.ndarray         # (n, 14) W m-2
    tsi: np.ndarray          # (n,) W m-2
    source: str
    sha256: str

    def __post_init__(self):
        n = self.days.size
        if self.days.ndim != 1 or self.band.shape != (n, NBANDS) or self.tsi.shape != (n,) or n < 1:
            raise SolarForcingError(f"{self.source}: SSI series needs matching days, (n, 14) bands and TSI")
        if not (np.all(np.isfinite(self.band)) and np.all(np.isfinite(self.tsi))):
            raise SolarForcingError(f"{self.source}: SSI series has non-finite values")
        if np.any(np.diff(self.days) <= 0.0):
            raise SolarForcingError(f"{self.source}: SSI series days are not strictly ascending")
        if np.any((self.tsi < 1300.0) | (self.tsi > 1400.0)):
            raise SolarForcingError(f"{self.source}: TSI outside 1300-1400 W m-2; is the unit W m-2?")
        if np.any(self.band <= 0.0):
            raise SolarForcingError(f"{self.source}: a day has a band with no irradiance; the series is damaged")
        # S_b / TSI, read at every radiation call.
        object.__setattr__(self, "fractions", self.band / self.tsi[:, None])

    def fractions_at(self, valid_time: datetime) -> np.ndarray:
        """S_b / TSI at ``valid_time`` (linear between daily noons, no extrapolation)."""
        day = (naive_utc(valid_time) - _EPOCH_1610).total_seconds() / 86400.0
        first, last = self.days[0], self.days[-1]
        if day < first - 0.5 or day > last + 0.5:
            raise SolarForcingError(
                f"{valid_time:%Y-%m-%d %H:%M} is outside the SSI series {self.source} "
                f"({_EPOCH_1610 + timedelta(days=first - 0.5):%Y-%m-%d} to "
                f"{_EPOCH_1610 + timedelta(days=last + 0.5):%Y-%m-%d}); extend it rather than extrapolate")
        frac = self.fractions
        if day <= first:
            return frac[0].copy()
        if day >= last:
            return frac[-1].copy()
        k = int(np.searchsorted(self.days, day, side="right")) - 1
        span = self.days[k + 1] - self.days[k]
        if span > 1.5:
            raise SolarForcingError(
                f"{valid_time:%Y-%m-%d %H:%M} falls in a {span:.0f}-day gap of the SSI series {self.source}")
        w = (day - self.days[k]) / span
        return frac[k] + w * (frac[k + 1] - frac[k])


def _band_weights(wavelength_nm, width_nm) -> np.ndarray:
    w = np.asarray(wavelength_nm, np.float64)
    dw = np.asarray(width_nm, np.float64)
    lo, hi = w - 0.5 * dw, w + 0.5 * dw
    weights = np.empty((NBANDS, w.size))
    for b, (a, z) in enumerate(BANDS_NM):
        weights[b] = np.clip(np.minimum(hi, z) - np.maximum(lo, a), 0.0, None)
    return weights


def read_netcdf(path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(noon days, (n, 14) band W m-2, TSI) from one NOAA CDR SSI netCDF.  A day
    is kept when its TSI and every bin inside RRTMG's bands are finite."""
    from woof.netcdf_bridge import open_dataset    # the Rust decoder; no Python NetCDF fallback
    with open_dataset(path) as data:
        need = ("SSI", "wavelength", "Wavelength_Band_Width", "TSI", "time")
        missing = [k for k in need if k not in data.variables]
        if missing:
            raise SolarForcingError(f"{path}: an SSI netCDF needs {need} (the NOAA CDR layout); missing {missing}")
        centre = cdr_day_centres(data, path)
        wl = np.asarray(data.variables["wavelength"][:], np.float64)
        dw = np.asarray(data.variables["Wavelength_Band_Width"][:], np.float64)
        ssi = np.ma.filled(np.ma.asarray(data.variables["SSI"][:]).astype(np.float64), np.nan)
        tsi = np.ma.filled(np.ma.asarray(data.variables["TSI"][:]).astype(np.float64), np.nan)
    if ssi.shape != (centre.size, wl.size):
        raise SolarForcingError(f"{path}: SSI of shape {ssi.shape}, expected {(centre.size, wl.size)}")
    weights = _band_weights(wl, dw)
    inside = weights.sum(axis=0) > 0.0
    ssi = ssi[:, inside]
    keep = np.isfinite(tsi) & np.all(np.isfinite(ssi), axis=1)
    return centre[keep], ssi[keep] @ weights[:, inside].T, tsi[keep]


def read_series(paths) -> SsiSeries:
    paths = [Path(p).expanduser() for p in (paths if isinstance(paths, (list, tuple)) else [paths])]
    days, bands, tsis, digest = [], [], [], hashlib.sha256()
    for p in paths:
        if not p.is_file():
            raise SolarForcingError(f"SSI file {p} does not exist")
        digest.update(p.read_bytes())
        d, b, t = read_netcdf(p)
        days.append(d)
        bands.append(b)
        tsis.append(t)
    d = np.concatenate(days)
    order = np.argsort(d, kind="stable")
    d = d[order]
    if np.any(np.diff(d) == 0.0):
        raise SolarForcingError(f"SSI files {[str(p) for p in paths]} repeat a day")
    return SsiSeries(d, np.concatenate(bands)[order], np.concatenate(tsis)[order],
                     ",".join(str(p) for p in paths), digest.hexdigest())


def fetch_nrlssi2(years, *, cache=None, opener=None) -> list[Path]:
    """The NOAA CDR daily SSI netCDF (about 13 MB) for each year, fetched once
    into the cache; the newest listed final (not preliminary) file is taken."""
    return fetch_cdr_years(
        years, prefix="ssi", index_url=NRLSSI2_SSI_INDEX_URL, subdir="nrlssi2-ssi",
        what="SSI", key="solar_ssi_source netCDF files in its layout",
        check=lambda path: SsiSeries(*read_netcdf(path), str(path), ""),
        cache=cache, opener=opener, timeout=300)


def load_series(source: str, start: datetime, end: datetime) -> SsiSeries:
    """The SSI series ``solar_ssi_source`` names (``nrlssi2`` or netCDF paths), covering start..end."""
    text = str(source or "").strip()
    if not text:
        raise SolarForcingError("solar_ssi_opt=1 needs solar_ssi_source: 'nrlssi2' or NOAA CDR SSI netCDF path(s)")
    if text == "nrlssi2":
        return read_series(fetch_nrlssi2(noon_years(start, end)))
    return read_series([p for p in text.split(os.pathsep) if p])


def band_factor(fractions, reference=None) -> np.ndarray:
    """f_b = (S_b/TSI) / (S_b,ref/TSI_ref), renormalised so that the in-band
    share of the TSI is the reference's; float32 (14,)."""
    ref = np.asarray(REF_BAND_FRACTION if reference is None else reference, np.float64)
    f = np.asarray(fractions, np.float64) / ref
    f *= ref.sum() / np.dot(ref, f)
    return f.astype(np.float32)


class SpectralVariability:
    """The per-radiation-call band factor (``solar_ssi_opt = 1``)."""

    def __init__(self, series: SsiSeries, start_time: datetime):
        self.series = series
        self.start_time = naive_utc(start_time)
        self.start_factor = self.factor(0.0)
        self.last_factor = self.start_factor

    @staticmethod
    def shortwave_targets(radiation) -> tuple:
        """The shortwave adapters inside ``radiation`` that take the band
        factor: composition leaves, and the adapter a wrapper (a partial nest's
        extended column) runs.  Resolved at each call, so a callable the
        driver swaps in (a batched ensemble's member radiation) is the one
        set.  Refused when no shortwave takes RRTMG's 14 bands."""
        from woof.core.radiation_composition import radiation_adapters
        found, leaves, seen = [], [], set()

        def visit(adapter):
            for leaf in radiation_adapters(adapter):
                if id(leaf) in seen:
                    continue
                seen.add(id(leaf))
                if getattr(type(leaf), "accepts_sw_band_factor", False):
                    if getattr(leaf, "shortwave", True):
                        found.append(leaf)
                elif "inner" in vars(leaf):          # a wrapper runs its own inner adapter
                    visit(vars(leaf)["inner"])
                else:
                    leaves.append(leaf)

        visit(radiation)
        if not found:
            names = sorted({type(a).__name__ for a in leaves})
            raise SolarForcingError(
                f"solar_ssi_opt=1 scales RRTMG SW's 14 bands, and this run's shortwave ({names}) "
                "does not take a band factor; it needs ra_sw_physics = 4 with ra_rrtmg_variant = "
                "'rrtmg_legacy', or 14 (RRTMG-K)")
        return tuple(found)

    @classmethod
    def for_run(cls, cfg, start_time, radiation=None):
        if not int(getattr(cfg, "solar_ssi_opt", 0) or 0):
            return None
        if start_time is None:
            raise SolarForcingError("solar_ssi_opt=1 needs the run's UTC start time")
        if radiation is not None:
            cls.shortwave_targets(radiation)          # refuse a shortwave without the bands now
        start = naive_utc(start_time)
        series = load_series(getattr(cfg, "solar_ssi_source", ""), start,
                             start + timedelta(seconds=float(cfg.run_seconds)))
        out = cls(series, start)       # refuses a series that misses the start now, not mid-run
        out.factor(float(cfg.run_seconds))
        return out

    def apply(self, radiation, elapsed_seconds: float) -> None:
        """Set the factor at this radiation call's time on every shortwave
        adapter ``radiation`` runs."""
        f = tuple(float(v) for v in self.factor(elapsed_seconds))
        for target in self.shortwave_targets(radiation):
            target.sw_band_factor = f

    def factor(self, elapsed_seconds: float) -> np.ndarray:
        frac = self.series.fractions_at(self.start_time + timedelta(seconds=float(elapsed_seconds)))
        self.last_factor = band_factor(frac)
        return self.last_factor

    def receipt(self) -> dict:
        return {"solar_ssi_source": self.series.source, "solar_ssi_sha256": self.series.sha256,
                "solar_ssi_band_factor_at_start": [round(float(v), 6) for v in self.start_factor]}


__all__ = ["BANDS_CM1", "BANDS_NM", "NBANDS", "REF_BAND_FRACTION", "SsiSeries", "SpectralVariability",
           "band_irradiance", "band_factor", "read_netcdf", "read_series", "fetch_nrlssi2", "load_series"]
