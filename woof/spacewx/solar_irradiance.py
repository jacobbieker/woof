"""Time-varying total solar irradiance (TSI) for the shortwave schemes.

WRF's radiation driver sets ``SOLCON = 1370 * ECCFAC`` (module_radiation_driver.F
``radconst``): a fixed 1370 W m-2 solar constant times the orbital eccentricity
factor.  The measured TSI is about 1361 W m-2 (SORCE/TIM, TSIS: Kopp & Lean 2011,
1360.8 +- 0.5 W m-2) and varies by about 1 W m-2 over the 11-year cycle, with
dips of several tenths of a percent when large sunspot groups cross the disc
(10-12 May 2024: 1360.4 W m-2).

``solar_tsi_opt = 1`` multiplies every shortwave output of a radiation call by a
scalar factor ``TSI(t) / reference``.  Shortwave radiative transfer is linear in
the top-of-atmosphere flux (RRTMG: ``solvar(ib) = scon / rrsw_scon`` multiplies
each band's source; Dudhia, Goddard, CAM, FLG and RRTMGP the same), so scaling
the outputs equals scaling SOLCON up to one float32 rounding, for every
shortwave scheme at once.  ``solar_tsi_mode``:

* ``"absolute"``: reference 1370 W m-2, i.e. SOLCON = TSI(t) * ECCFAC; replaces
  WRF's constant by the measured irradiance (about -0.6 %).
* ``"relative"``: reference 1360.8 W m-2 (the TIM cycle mean), keeping WRF's
  1370 calibration and adding only the solar variability.

The TSI series is daily (NOAA Climate Data Record, NRLSSI2 v03; or a user
file); values are daily means placed at noon UTC and interpolated linearly.  A
time outside the series is refused rather than extrapolated.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import os
from pathlib import Path
import re

import numpy as np

#: WRF's radconst solar constant (module_radiation_driver.F), W m-2.
WRF_SOLAR_CONSTANT = 1370.0
#: The solar-cycle mean TSI (Kopp & Lean 2011, GRL 38, L01706), W m-2.
TSI_CYCLE_MEAN = 1360.8
MODES = ("absolute", "relative")

#: NOAA CDR daily TSI (NRLSSI2 v03), one netCDF per year.
NRLSSI2_INDEX_URL = "https://www.ncei.noaa.gov/data/total-solar-irradiance/access/daily/"
ENV_CACHE = "WOOF_SPACEWX_CACHE"
_EPOCH_1610 = datetime(1610, 1, 1)


class SolarForcingError(ValueError):
    pass


def cache_root() -> Path:
    """``~/.woof/spacewx`` (or ``$WOOF_SPACEWX_CACHE``): fetched index data."""
    explicit = os.environ.get(ENV_CACHE, "").strip()
    return Path(explicit).expanduser() if explicit else Path.home() / ".woof" / "spacewx"


def naive_utc(value: datetime) -> datetime:
    """``value`` as a naive UTC datetime: an aware one is converted, not relabelled."""
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


_naive_utc = naive_utc


@dataclass(frozen=True)
class TsiSeries:
    """Daily TSI means (W m-2) at noon UTC of each day, ascending, no gaps over 1 day."""

    days: np.ndarray        # float64 days since 1610-01-01 00 UTC of each daily mean's noon
    tsi: np.ndarray         # float64 W m-2
    source: str
    sha256: str

    def __post_init__(self):
        if self.days.ndim != 1 or self.days.shape != self.tsi.shape or self.days.size < 1:
            raise SolarForcingError(f"{self.source}: TSI series needs matching 1-D days and values")
        if not (np.all(np.isfinite(self.tsi)) and np.all(np.isfinite(self.days))):
            raise SolarForcingError(f"{self.source}: TSI series has non-finite values")
        if np.any(np.diff(self.days) <= 0.0):
            raise SolarForcingError(f"{self.source}: TSI series days are not strictly ascending")
        if np.any((self.tsi < 1300.0) | (self.tsi > 1400.0)):
            raise SolarForcingError(f"{self.source}: TSI outside 1300-1400 W m-2; is the unit W m-2?")

    def at(self, valid_time: datetime) -> float:
        """TSI at ``valid_time``: linear between daily noons; refuses gaps and extrapolation.

        The half day before the first noon and after the last noon takes that day's mean."""
        day = (_naive_utc(valid_time) - _EPOCH_1610).total_seconds() / 86400.0
        first, last = self.days[0], self.days[-1]
        if day < first - 0.5 or day > last + 0.5:
            raise SolarForcingError(
                f"{valid_time:%Y-%m-%d %H:%M} is outside the TSI series {self.source} "
                f"({_EPOCH_1610 + timedelta(days=first - 0.5):%Y-%m-%d} to "
                f"{_EPOCH_1610 + timedelta(days=last + 0.5):%Y-%m-%d}); extend it rather than extrapolate")
        if day <= first:
            return float(self.tsi[0])
        if day >= last:
            return float(self.tsi[-1])
        k = int(np.searchsorted(self.days, day, side="right")) - 1
        span = self.days[k + 1] - self.days[k]
        if span > 1.5:
            raise SolarForcingError(
                f"{valid_time:%Y-%m-%d %H:%M} falls in a {span:.0f}-day gap of the TSI series {self.source}")
        w = (day - self.days[k]) / span
        return float(self.tsi[k] + w * (self.tsi[k + 1] - self.tsi[k]))


def _read_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    days, values = [], []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        text = line.split("#", 1)[0].strip()
        if not text:
            continue
        fields = [f.strip() for f in re.split(r"[,\s]+", text) if f.strip()]
        if len(fields) != 2:
            raise SolarForcingError(f"{path}:{number}: expected 'YYYY-MM-DD,TSI_Wm2', got {line!r}")
        try:
            date = datetime.strptime(fields[0], "%Y-%m-%d")
            value = float(fields[1])
        except ValueError as error:
            if not days and number == 1:
                continue        # one header line
            raise SolarForcingError(f"{path}:{number}: {error}") from error
        days.append((date - _EPOCH_1610).total_seconds() / 86400.0 + 0.5)
        values.append(value)
    return np.asarray(days, np.float64), np.asarray(values, np.float64)


def cdr_day_centres(data, path) -> np.ndarray:
    """The centres of a NOAA CDR file's daily means, in days since 1610-01-01."""
    units = str(getattr(data.variables["time"], "units", ""))
    if units.strip() != "days since 1610-01-01 00:00:00":
        raise SolarForcingError(f"{path}: time units {units!r}; expected the CDR's 'days since 1610-01-01 00:00:00'")
    start = np.asarray(data.variables["time"][:], np.float64)
    if "time_bnds" in data.variables:
        bounds = np.asarray(data.variables["time_bnds"][:], np.float64)
        return 0.5 * (bounds[:, 0] + bounds[:, 1])
    return start + 0.5


def _read_netcdf(path: Path) -> tuple[np.ndarray, np.ndarray]:
    from woof.netcdf_bridge import open_dataset    # the Rust decoder; no Python NetCDF fallback
    with open_dataset(path) as data:
        if "TSI" not in data.variables or "time" not in data.variables:
            raise SolarForcingError(f"{path}: a TSI netCDF needs 'TSI' and 'time' (the NOAA CDR layout)")
        centre = cdr_day_centres(data, path)
        values = np.ma.filled(np.ma.asarray(data.variables["TSI"][:]).astype(np.float64), np.nan)
    keep = np.isfinite(values)
    return centre[keep], values[keep]


def read_series(path) -> TsiSeries:
    """One or more TSI files (CSV ``YYYY-MM-DD,TSI`` or NOAA CDR netCDF), merged in time order."""
    paths = [Path(p).expanduser() for p in (path if isinstance(path, (list, tuple)) else [path])]
    days, values, digest = [], [], hashlib.sha256()
    for p in paths:
        if not p.is_file():
            raise SolarForcingError(f"TSI file {p} does not exist")
        digest.update(p.read_bytes())
        d, v = (_read_netcdf(p) if p.suffix in (".nc", ".nc4") else _read_csv(p))
        days.append(d)
        values.append(v)
    d = np.concatenate(days)
    v = np.concatenate(values)
    order = np.argsort(d, kind="stable")
    d, v = d[order], v[order]
    if np.any(np.diff(d) == 0.0):
        raise SolarForcingError(f"TSI files {[str(p) for p in paths]} repeat a day")
    return TsiSeries(d, v, ",".join(str(p) for p in paths), digest.hexdigest())


def fetch_nrlssi2(years, *, cache=None, opener=None) -> list[Path]:
    """The NOAA CDR daily TSI netCDF for each year, fetched once into the cache.

    The CDR republishes a year's file with a new creation stamp (``_cYYYYMMDD``);
    the newest listed one is taken and its SHA-256 goes into the run's receipt
    through :class:`TsiSeries` (printed by the driver as the
    ``[spacewx tsi]`` start-up line).  NOAA CDR data are open (cite Coddington et al.
    2016, BAMS 97, and the CDR DOI 10.7289/V56W985W)."""
    return fetch_cdr_years(
        years, prefix="tsi", index_url=NRLSSI2_INDEX_URL,
        subdir="nrlssi2-tsi", what="TSI", key="solar_tsi_source a CSV or netCDF",
        check=lambda path: TsiSeries(*_read_netcdf(path), str(path), ""),
        cache=cache, opener=opener, timeout=120)


def fetch_cdr_years(years, *, prefix, index_url, subdir, what, key, check,
                    cache=None, opener=None, timeout=120) -> list[Path]:
    """The NOAA CDR daily ``{prefix}_v03r00_daily_s{Y}0101_e{Y}1231_c*.nc`` file
    of each year (the newest final one listed; preliminary files are not
    taken), fetched once into ``cache_root()/subdir``.  ``check(path)`` must
    accept a download before it is installed."""
    import urllib.request
    opener = opener or (lambda url: urllib.request.urlopen(url, timeout=timeout))
    root = Path(cache) if cache is not None else cache_root() / subdir
    root.mkdir(parents=True, exist_ok=True)
    listing = None
    out = []
    for year in sorted(set(int(y) for y in years)):
        pattern = f"{prefix}_v03r00_daily_s{year}0101_e{year}1231_c"
        local = sorted(root.glob(pattern + "*.nc"))
        if local:
            out.append(local[-1])
            continue
        if listing is None:
            with opener(index_url) as reply:
                listing = reply.read().decode("utf-8", "replace")
        names = sorted(set(re.findall(re.escape(pattern) + r"\d{8}\.nc", listing)))
        if not names:
            raise SolarForcingError(
                f"no complete NOAA CDR {what} file for {year} at {index_url} (the CDR lags by a few "
                f"months); give {key} that covers the run")
        target = root / names[-1]
        from woof.spacewx.assets import unique_part
        part = unique_part(target)
        try:
            with opener(index_url + names[-1]) as reply, part.open("wb") as stream:
                stream.write(reply.read())
        except BaseException:
            part.unlink(missing_ok=True)
            raise
        from woof.netcdf_bridge import NetcdfBridgeMissing
        try:
            # A truncated or wrong download must not be installed: it must read
            # as a series (decodes, finite, ascending, in W m-2) first.
            check(part)
        except NetcdfBridgeMissing:
            part.unlink(missing_ok=True)      # the reader is missing, not the file bad
            raise
        except Exception as exc:
            part.unlink(missing_ok=True)
            raise SolarForcingError(f"{index_url}{names[-1]}: the download does not read "
                                    f"as a NOAA CDR {what} file ({exc}); try again") from exc
        except BaseException:
            part.unlink(missing_ok=True)
            raise
        part.replace(target)
        out.append(target)
    return out


def noon_years(start: datetime, end: datetime) -> range:
    """The years of the daily means (at noon) a run from start to end reads:
    a time is covered by the days within half a day of it."""
    start, end = _naive_utc(start), _naive_utc(end)
    return range((start - timedelta(hours=12)).year, (end + timedelta(hours=12)).year + 1)


def load_series(source: str, start: datetime, end: datetime) -> TsiSeries:
    """The TSI series ``solar_tsi_source`` names, covering ``start``..``end``."""
    text = str(source or "").strip()
    if not text:
        raise SolarForcingError("solar_tsi_opt=1 needs solar_tsi_source: a TSI CSV/netCDF path or 'nrlssi2'")
    if text == "nrlssi2":
        return read_series(fetch_nrlssi2(noon_years(start, end)))
    return read_series([p for p in text.split(os.pathsep) if p])


def reference(mode: str) -> float:
    if mode not in MODES:
        raise SolarForcingError(f"solar_tsi_mode={mode!r} must be one of {MODES}")
    return WRF_SOLAR_CONSTANT if mode == "absolute" else TSI_CYCLE_MEAN


class SolarVariability:
    """The per-radiation-call TSI factor and the shortwave scaling it applies."""

    #: RadiationResult boundary/diagnostic flux names that are shortwave.
    @staticmethod
    def is_shortwave(name: str) -> bool:
        return name.lower().startswith("sw") or "SW" in name

    def __init__(self, series: TsiSeries, start_time: datetime, mode: str = "absolute"):
        self.series = series
        self.start_time = _naive_utc(start_time)
        self.mode = mode
        self.reference = reference(mode)
        self.last_tsi = None
        self.last_factor = None

    @classmethod
    def for_run(cls, cfg, start_time):
        if not int(getattr(cfg, "solar_tsi_opt", 0) or 0):
            return None
        if start_time is None:
            raise SolarForcingError("solar_tsi_opt=1 needs the run's UTC start time")
        start = _naive_utc(start_time)
        series = load_series(getattr(cfg, "solar_tsi_source", ""), start,
                             start + timedelta(seconds=float(cfg.run_seconds)))
        out = cls(series, start, getattr(cfg, "solar_tsi_mode", "absolute"))
        out.factor(0.0)       # refuse a series that misses the start now, not mid-run
        out.factor(float(cfg.run_seconds))
        return out

    def factor(self, elapsed_seconds: float) -> float:
        tsi = self.series.at(self.start_time + timedelta(seconds=float(elapsed_seconds)))
        self.last_tsi = tsi
        self.last_factor = tsi / self.reference
        return self.last_factor

    def scale(self, result, elapsed_seconds: float):
        """Scale every shortwave output of one RadiationResult in place by TSI(t)/reference."""
        f = np.float32(self.factor(elapsed_seconds))

        def mul(array):
            # float32 array * float32 scalar: one rounding, the dtype kept (NumPy or CuPy).
            return None if array is None else array * f

        sw_old = result.rthratensw
        result.rthratensw = mul(sw_old)
        if result.rthraten_total is not None:
            # total = LW + SW: move only the shortwave part.
            result.rthraten_total = result.rthraten_total + (result.rthratensw - sw_old)
        for name in ("swdown", "gsw", "swddir", "swddif"):
            setattr(result, name, mul(getattr(result, name)))
        for attr in ("boundary_fluxes", "flux_diagnostics", "spectral_fluxes", "spectral_surface"):
            mapping = getattr(result, attr)
            if mapping:
                setattr(result, attr, {k: (mul(v) if self.is_shortwave(k) else v) for k, v in mapping.items()})
        return result

    #: Shortwave outputs a scheme writes straight into the driver's fields
    #: rather than returning on RadiationResult, by ra_sw_physics.
    DIRECT_SW_FIELDS = {2: ("rswtoa",), 99: ("rswtoa", "swdownc")}

    def scale_direct_fields(self, fields, ra_sw_physics: int) -> None:
        """Scale, in place and by the factor :meth:`scale` just applied, the
        shortwave outputs the scheme ``ra_sw_physics`` wrote into ``fields``."""
        f = np.float32(self.last_factor)
        for name in self.DIRECT_SW_FIELDS.get(int(ra_sw_physics), ()):
            if name in fields:
                fields[name] *= f

    def receipt(self) -> dict:
        return {"solar_tsi_source": self.series.source, "solar_tsi_sha256": self.series.sha256,
                "solar_tsi_mode": self.mode, "solar_tsi_reference_wm2": self.reference}
