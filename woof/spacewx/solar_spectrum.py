"""Solar spectral irradiance (SSI) variability for RRTMG's shortwave bands.

``solar_tsi_opt`` scales every shortwave output by TSI(t) / reference: the
spectrum keeps RRTMG's fixed shape (Kurucz, in ``sfluxref``).  Over the solar
cycle the ultraviolet varies far more than the visible (200-260 nm by several
per cent, 400-700 nm by about 0.1 %), so the shape changes too.
``solar_ssi_opt = 1`` multiplies each of RRTMG SW's 14 bands (16-29) by

    f_b(t) = [S_b(t) / TSI(t)] / [S_b,ref / TSI_ref]

renormalised so that sum_b w_b f_b = sum_b w_b with w_b = S_b,ref / TSI_ref:
the band shares move, and the total stays the TSI factor's.  S_b is the NOAA
CDR SSI v03 daily SSI (the NNLSSI model; ``solar_ssi_source = 'nrlssi2'`` keeps
its historical name) integrated over the band; the reference is the mean over
solar cycle 24 (2009-2019), shipped as :data:`REF_BAND_FRACTION` and refitted
by ``tools/spacewx_ssi_reference.py``.  RRTMG's own band weights (Kurucz)
are within 1.2 % of the CDR's except 263-345 nm (+7 %), so the total of RRTMG's
source moves by sum_b (wK_b - w_b) (f_b - 1), about 1e-5 of the TSI.

The factor reaches the scheme as ``band_factor`` on
``rrtmg_sw_batched_device`` (it multiplies ``adjflux``), for RRTMG
(``ra_sw_physics = 4``) and RRTMG-K (14).  Daily means are placed at noon UTC
and interpolated linearly; a time outside the series is refused.

Photolysis takes the same spectrum on its own wavelength bins
(:class:`PhotolysisSpectrum`): each bin's extraterrestrial flux is multiplied
by [S_bin(t) / TSI(t)] / [S_bin,ref / TSI_ref], with the reference integrated
from the shipped cycle-24 mean spectrum (``woof/data/spacewx/
ssi_reference_v03_2009_2019.json``, the CDR's bins below 760 nm, the same
days as :data:`REF_BAND_FRACTION`).  These bins cover only part of the
spectrum, so they are not renormalised: the ultraviolet's share of the TSI
moves as the CDR says, and the TSI itself stays whatever the photolysis
scheme's own constant flux implies.
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
#: S_b,ref / TSI_ref: the CDR SSI v03 (NNLSSI) band shares averaged over 2009-01-01 to
#: 2019-12-31 (tools/spacewx_ssi_reference.py; NOAA CDR, open data).
REF_BAND_FRACTION = (0.00884400, 0.01482416, 0.01726081, 0.01624697, 0.04018645, 0.07476922, 0.01767045,
                     0.25270815, 0.15969940, 0.25604600, 0.09551557, 0.03421286, 0.00219724, 0.00940440)
#: NOAA CDR daily SSI (v03, NNLSSI, standard resolution), one netCDF per year.
NRLSSI2_SSI_INDEX_URL = ("https://www.ncei.noaa.gov/data/solar-spectral-irradiance/access/"
                         "standard-resolution/daily/")
SOURCES = ("nrlssi2",)


def band_irradiance(wavelength_nm, width_nm, ssi, bands_nm=BANDS_NM) -> np.ndarray:
    """(..., nbands) W m-2 in ``bands_nm`` (RRTMG's by default) from SSI (..., n)
    in W m-2 nm-1 on bins centred at ``wavelength_nm`` of width ``width_nm``
    (partial bins pro rata)."""
    return np.asarray(ssi, np.float64) @ _band_weights(wavelength_nm, width_nm, bands_nm).T


@dataclass(frozen=True)
class SsiSeries:
    """Daily band irradiances (W m-2) and TSI at noon UTC of each day, ascending."""

    days: np.ndarray         # float64 days since 1610-01-01 00 UTC
    band: np.ndarray         # (n, nbands) W m-2
    tsi: np.ndarray          # (n,) W m-2
    source: str
    sha256: str
    bands_nm: tuple = BANDS_NM   # the bands' (low, high) edges, nm

    def __post_init__(self):
        n, nb = self.days.size, len(self.bands_nm)
        if self.days.ndim != 1 or self.band.shape != (n, nb) or self.tsi.shape != (n,) or n < 1:
            raise SolarForcingError(f"{self.source}: SSI series needs matching days, (n, {nb}) bands and TSI")
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


def _band_weights(wavelength_nm, width_nm, bands_nm=BANDS_NM) -> np.ndarray:
    w = np.asarray(wavelength_nm, np.float64)
    dw = np.asarray(width_nm, np.float64)
    lo, hi = w - 0.5 * dw, w + 0.5 * dw
    weights = np.empty((len(bands_nm), w.size))
    for b, (a, z) in enumerate(bands_nm):
        weights[b] = np.clip(np.minimum(hi, z) - np.maximum(lo, a), 0.0, None)
    return weights


def read_netcdf(path, bands_nm=BANDS_NM) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(noon days, (n, nbands) band W m-2, TSI) from one NOAA CDR SSI netCDF.  A
    day is kept when its TSI and every bin inside the bands (RRTMG's by default)
    are finite."""
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
    weights = _band_weights(wl, dw, bands_nm)
    inside = weights.sum(axis=0) > 0.0
    ssi = ssi[:, inside]
    keep = np.isfinite(tsi) & np.all(np.isfinite(ssi), axis=1)
    return centre[keep], ssi[keep] @ weights[:, inside].T, tsi[keep]


def read_series(paths, bands_nm=BANDS_NM) -> SsiSeries:
    paths = [Path(p).expanduser() for p in (paths if isinstance(paths, (list, tuple)) else [paths])]
    days, bands, tsis, digest = [], [], [], hashlib.sha256()
    for p in paths:
        if not p.is_file():
            raise SolarForcingError(f"SSI file {p} does not exist")
        digest.update(p.read_bytes())
        d, b, t = read_netcdf(p, bands_nm)
        days.append(d)
        bands.append(b)
        tsis.append(t)
    d = np.concatenate(days)
    order = np.argsort(d, kind="stable")
    d = d[order]
    if np.any(np.diff(d) == 0.0):
        raise SolarForcingError(f"SSI files {[str(p) for p in paths]} repeat a day")
    return SsiSeries(d, np.concatenate(bands)[order], np.concatenate(tsis)[order],
                     ",".join(str(p) for p in paths), digest.hexdigest(), tuple(bands_nm))


def fetch_nrlssi2(years, *, cache=None, opener=None) -> list[Path]:
    """The NOAA CDR daily SSI netCDF (about 13 MB) for each year, fetched once
    into the cache; the newest listed final (not preliminary) file is taken."""
    return fetch_cdr_years(
        years, prefix="ssi", index_url=NRLSSI2_SSI_INDEX_URL, subdir="nrlssi2-ssi",
        what="SSI", key="solar_ssi_source netCDF files in its layout",
        check=lambda path: SsiSeries(*read_netcdf(path), str(path), ""),
        cache=cache, opener=opener, timeout=300)


def load_series(source: str, start: datetime, end: datetime, bands_nm=BANDS_NM) -> SsiSeries:
    """The SSI series ``solar_ssi_source`` names (``nrlssi2`` or netCDF paths), covering start..end."""
    text = str(source or "").strip()
    if not text:
        raise SolarForcingError("solar_ssi_opt=1 needs solar_ssi_source: 'nrlssi2' or NOAA CDR SSI netCDF path(s)")
    if text == "nrlssi2":
        return read_series(fetch_nrlssi2(noon_years(start, end)), bands_nm)
    return read_series([p for p in text.split(os.pathsep) if p], bands_nm)


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


#: The shipped cycle-24 mean spectrum S(lambda) / TSI (nm-1) below 760 nm.
REFERENCE_SPECTRUM = "woof/data/spacewx/ssi_reference_v03_2009_2019.json"
_REFERENCE = {}


def reference_spectrum() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(wavelength nm, width nm, S/TSI nm-1) of :data:`REFERENCE_SPECTRUM`."""
    if not _REFERENCE:
        import json
        path = Path(__file__).resolve().parents[1] / "data" / "spacewx" / Path(REFERENCE_SPECTRUM).name
        doc = json.loads(path.read_text(encoding="utf-8"))
        _REFERENCE["v"] = tuple(np.asarray(doc[k], np.float64)
                                for k in ("wavelength_nm", "width_nm", "fraction_per_nm"))
    return _REFERENCE["v"]


def reference_fractions(bands_nm) -> np.ndarray:
    """S_bin,ref / TSI_ref for each (low, high) nm bin, from the shipped spectrum;
    refused for a bin reaching past its 760 nm end or with no reference energy."""
    wl, dw, frac = reference_spectrum()
    top = float(wl[-1] + 0.5 * dw[-1])
    for lo, hi in bands_nm:
        if not (0.0 <= lo < hi <= top):
            raise SolarForcingError(f"solar_ssi_opt: bin {lo:g}-{hi:g} nm is outside the reference spectrum "
                                    f"(0-{top:g} nm)")
    out = _band_weights(wl, dw, bands_nm) @ frac
    if np.any(out <= 0.0):
        raise SolarForcingError("solar_ssi_opt: a bin has no reference energy")
    return out


def bin_factor(fractions, bands_nm) -> np.ndarray:
    """f_bin = (S_bin/TSI) / (S_bin,ref/TSI_ref), not renormalised (the bins
    cover part of the spectrum); float64."""
    return np.asarray(fractions, np.float64) / reference_fractions(bands_nm)


class PhotolysisSpectrum:
    """The per-photolysis-call factor on a scheme's extraterrestrial-flux bins
    (``solar_ssi_opt = 1`` with a photolysis scheme)."""

    def __init__(self, series: SsiSeries, start_time: datetime, scheme: str):
        self.series = series
        self.bands_nm = tuple(series.bands_nm)
        self.scheme = scheme
        self.start_time = naive_utc(start_time)
        reference_fractions(self.bands_nm)            # refuse bins the reference cannot cover now
        self.start_factor = self.factor(0.0)

    @classmethod
    def for_run(cls, cfg, start_time, bands_nm, scheme: str):
        if not int(getattr(cfg, "solar_ssi_opt", 0) or 0):
            return None
        if start_time is None:
            raise SolarForcingError("solar_ssi_opt=1 needs the run's UTC start time")
        start = naive_utc(start_time)
        series = load_series(getattr(cfg, "solar_ssi_source", ""), start,
                             start + timedelta(seconds=float(cfg.run_seconds)), bands_nm)
        out = cls(series, start, scheme)
        out.factor(float(cfg.run_seconds))
        return out

    def factor(self, elapsed_seconds: float) -> np.ndarray:
        """The bins' factors at start + ``elapsed_seconds`` (float64)."""
        frac = self.series.fractions_at(self.start_time + timedelta(seconds=float(elapsed_seconds)))
        return bin_factor(frac, self.bands_nm)

    def receipt(self) -> dict:
        return {f"solar_ssi_{self.scheme}_factor_at_start": [round(float(v), 6) for v in self.start_factor]}


__all__ = ["BANDS_CM1", "BANDS_NM", "NBANDS", "REF_BAND_FRACTION", "SsiSeries", "SpectralVariability",
           "band_irradiance", "band_factor", "read_netcdf", "read_series", "fetch_nrlssi2", "load_series",
           "REFERENCE_SPECTRUM", "reference_spectrum", "reference_fractions", "bin_factor", "PhotolysisSpectrum"]
