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
import re

import numpy as np

from woof.spacewx.solar_irradiance import _EPOCH_1610, SolarForcingError, cache_root, naive_utc

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
    w = np.asarray(wavelength_nm, np.float64)
    dw = np.asarray(width_nm, np.float64)
    lo, hi = w - 0.5 * dw, w + 0.5 * dw
    weights = np.empty((NBANDS, w.size))
    for b, (a, z) in enumerate(BANDS_NM):
        weights[b] = np.clip(np.minimum(hi, z) - np.maximum(lo, a), 0.0, None)
    return np.asarray(ssi, np.float64) @ weights.T


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

    def fractions_at(self, valid_time: datetime) -> np.ndarray:
        """S_b / TSI at ``valid_time`` (linear between daily noons, no extrapolation)."""
        day = (naive_utc(valid_time) - _EPOCH_1610).total_seconds() / 86400.0
        first, last = self.days[0], self.days[-1]
        if day < first - 0.5 or day > last + 0.5:
            raise SolarForcingError(
                f"{valid_time:%Y-%m-%d %H:%M} is outside the SSI series {self.source} "
                f"({_EPOCH_1610 + timedelta(days=first - 0.5):%Y-%m-%d} to "
                f"{_EPOCH_1610 + timedelta(days=last + 0.5):%Y-%m-%d}); extend it rather than extrapolate")
        frac = self.band / self.tsi[:, None]
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


def read_netcdf(path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(noon days, (n, 14) band W m-2, TSI) from one NOAA CDR SSI netCDF."""
    from woof.netcdf_bridge import open_dataset    # the Rust decoder; no Python NetCDF fallback
    with open_dataset(path) as data:
        need = ("SSI", "wavelength", "Wavelength_Band_Width", "TSI", "time")
        missing = [k for k in need if k not in data.variables]
        if missing:
            raise SolarForcingError(f"{path}: an SSI netCDF needs {need} (the NOAA CDR layout); missing {missing}")
        units = str(getattr(data.variables["time"], "units", ""))
        if units.strip() != "days since 1610-01-01 00:00:00":
            raise SolarForcingError(f"{path}: time units {units!r}; expected 'days since 1610-01-01 00:00:00'")
        start = np.asarray(data.variables["time"][:], np.float64)
        if "time_bnds" in data.variables:
            bounds = np.asarray(data.variables["time_bnds"][:], np.float64)
            centre = 0.5 * (bounds[:, 0] + bounds[:, 1])
        else:
            centre = start + 0.5
        wl = np.asarray(data.variables["wavelength"][:], np.float64)
        dw = np.asarray(data.variables["Wavelength_Band_Width"][:], np.float64)
        ssi = np.ma.filled(np.ma.asarray(data.variables["SSI"][:]).astype(np.float64), np.nan)
        tsi = np.ma.filled(np.ma.asarray(data.variables["TSI"][:]).astype(np.float64), np.nan)
    if ssi.shape != (centre.size, wl.size):
        raise SolarForcingError(f"{path}: SSI of shape {ssi.shape}, expected {(centre.size, wl.size)}")
    keep = np.isfinite(tsi) & np.all(np.isfinite(ssi), axis=1)
    return centre[keep], band_irradiance(wl, dw, ssi[keep]), tsi[keep]


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
    import urllib.request
    opener = opener or (lambda url: urllib.request.urlopen(url, timeout=300))
    root = Path(cache) if cache is not None else cache_root() / "nrlssi2-ssi"
    root.mkdir(parents=True, exist_ok=True)
    listing = None
    out = []
    for year in sorted(set(int(y) for y in years)):
        local = sorted(root.glob(f"ssi_v03r00_daily_s{year}0101_e{year}1231_c*.nc"))
        if local:
            out.append(local[-1])
            continue
        if listing is None:
            with opener(NRLSSI2_SSI_INDEX_URL) as reply:
                listing = reply.read().decode("utf-8", "replace")
        names = sorted(set(re.findall(rf'ssi_v03r00_daily_s{year}0101_e{year}1231_c\d{{8}}\.nc', listing)))
        if not names:
            raise SolarForcingError(
                f"no complete NOAA CDR SSI file for {year} at {NRLSSI2_SSI_INDEX_URL} (the CDR lags by a "
                "few months); give solar_ssi_source netCDF files in its layout that cover the run")
        target = root / names[-1]
        from woof.spacewx.assets import unique_part
        part = unique_part(target)
        try:
            with opener(NRLSSI2_SSI_INDEX_URL + names[-1]) as reply, part.open("wb") as stream:
                stream.write(reply.read())
        except BaseException:
            part.unlink(missing_ok=True)
            raise
        from woof.netcdf_bridge import NetcdfBridgeMissing
        try:
            SsiSeries(*read_netcdf(part), str(part), "")      # a truncated download is not installed
        except NetcdfBridgeMissing:
            part.unlink(missing_ok=True)
            raise
        except Exception as exc:
            part.unlink(missing_ok=True)
            raise SolarForcingError(f"{NRLSSI2_SSI_INDEX_URL}{names[-1]}: the download does not read "
                                    f"as a NOAA CDR SSI file ({exc}); try again") from exc
        except BaseException:
            part.unlink(missing_ok=True)
            raise
        part.replace(target)
        out.append(target)
    return out


def load_series(source: str, start: datetime, end: datetime) -> SsiSeries:
    """The SSI series ``solar_ssi_source`` names (``nrlssi2`` or netCDF paths), covering start..end."""
    text = str(source or "").strip()
    if not text:
        raise SolarForcingError("solar_ssi_opt=1 needs solar_ssi_source: 'nrlssi2' or NOAA CDR SSI netCDF path(s)")
    if text == "nrlssi2":
        start, end = naive_utc(start), naive_utc(end)
        years = range((start - timedelta(hours=12)).year, (end + timedelta(hours=12)).year + 1)
        return read_series(fetch_nrlssi2(years))
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

    #: SW schemes whose source is RRTMG's 14 bands (RRTMG, RRTMG-K).
    SCHEMES = (4, 14)

    def __init__(self, series: SsiSeries, start_time: datetime, targets=()):
        self.series = series
        self.start_time = naive_utc(start_time)
        self.targets = tuple(targets)
        self.last_factor = None

    @staticmethod
    def shortwave_targets(radiation) -> tuple:
        """The shortwave leaves of ``radiation`` that take the band factor;
        refused when the shortwave is not RRTMG's 14 bands."""
        from woof.core.radiation_composition import radiation_adapters
        leaves = [a for a in radiation_adapters(radiation) if getattr(a, "shortwave", True)]
        targets = tuple(a for a in leaves if getattr(a, "accepts_sw_band_factor", False))
        if not targets:
            names = sorted({type(a).__name__ for a in leaves})
            raise SolarForcingError(
                f"solar_ssi_opt=1 scales RRTMG SW's 14 bands, and this run's shortwave ({names}) "
                "does not take a band factor; it needs ra_sw_physics = 4 with ra_rrtmg_variant = "
                "'rrtmg_legacy', or 14 (RRTMG-K)")
        return targets

    @classmethod
    def for_run(cls, cfg, start_time, radiation=None):
        if not int(getattr(cfg, "solar_ssi_opt", 0) or 0):
            return None
        if start_time is None:
            raise SolarForcingError("solar_ssi_opt=1 needs the run's UTC start time")
        targets = cls.shortwave_targets(radiation) if radiation is not None else ()
        start = naive_utc(start_time)
        series = load_series(getattr(cfg, "solar_ssi_source", ""), start,
                             start + timedelta(seconds=float(cfg.run_seconds)))
        out = cls(series, start, targets)
        out.factor(0.0)       # refuse a series that misses the start now, not mid-run
        out.factor(float(cfg.run_seconds))
        return out

    def apply(self, elapsed_seconds: float) -> None:
        """Set the factor at this radiation call's time on every target leaf."""
        f = tuple(float(v) for v in self.factor(elapsed_seconds))
        for target in self.targets:
            target.sw_band_factor = f

    def factor(self, elapsed_seconds: float) -> np.ndarray:
        frac = self.series.fractions_at(self.start_time + timedelta(seconds=float(elapsed_seconds)))
        self.last_factor = band_factor(frac)
        return self.last_factor

    def receipt(self) -> dict:
        f = self.factor(0.0)
        return {"solar_ssi_source": self.series.source, "solar_ssi_sha256": self.series.sha256,
                "solar_ssi_band_factor_at_start": [round(float(v), 6) for v in f]}


__all__ = ["BANDS_CM1", "BANDS_NM", "NBANDS", "REF_BAND_FRACTION", "SsiSeries", "SpectralVariability",
           "band_irradiance", "band_factor", "read_netcdf", "read_series", "fetch_nrlssi2", "load_series"]
