"""Solar wind and interplanetary magnetic field: readers and coupling functions.

Sources, both public and both fetched on demand into the space-weather cache:

* ``"omni"``: NASA SPDF's hourly OMNI2 files (King & Papitashvili 2005, JGR 110,
  A02104; ``spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_YYYY.dat``),
  time-shifted to the bow-shock nose.  Words used (omni2.text): 16-17 By/Bz GSM
  (nT), 24 proton density (cm-3), 25 flow speed (km/s), 39 Kp (x10, from GFZ),
  41 Dst (nT, Kyoto), 43-48 the integral proton flux above 1, 2, 4, 10, 30 and
  60 MeV (cm-2 s-1 sr-1, IMP and GOES), 52 PC(N) (the northern polar cap
  index, DTU) and 53 AL (nT, Kyoto).
* ``"swpc"``: NOAA SWPC's real-time solar wind (``json/rtsw/rtsw_mag_1m.json``
  and ``rtsw_wind_1m.json``, the last 24 hours at 1 minute from the active L1
  spacecraft; public domain), averaged to hourly means.  It is measured at L1,
  roughly 30-60 minutes upstream, and is not time-shifted.

A local OMNI2-format file (or several, joined by ``:``) can stand in for the
download.

Coupling functions:

* Newell et al. 2007 (JGR 112, A01206):
  ``dPhi/dt = v^(4/3) B_T^(2/3) sin^(8/3)(theta_c / 2)``, with
  ``B_T = sqrt(By^2 + Bz^2)`` and ``theta_c = atan2(By, Bz)`` in GSM.
* Boyle, Reiff & Hairston 1997 (JGR 102, 111), the polar cap potential:
  ``Phi = 1e-4 v^2 + 11.7 B sin^3(theta / 2)`` kV, ``theta = arccos(Bz / B)``.
* A Kp estimate from the solar wind (:func:`kp_estimate`), fitted here (see
  :data:`KP_FIT`) because the published Kp regressions are not all
  reproducible from their papers.  Each 3-hour Kp interval starting at T is
  regressed on the mean of the hourly values stamped T-1, T and T+1:
  ``Kp = a + b (1e-4 dPhi/dt)^p + c (1e-7 sqrt(n) v^2)``, clipped to [0, 9].
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from woof.spacewx.assets import unique_part
from woof.spacewx.solar_irradiance import cache_root, naive_utc as _naive_utc

OMNI_URL = "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_{year}.dat"
SWPC_MAG_URL = "https://services.swpc.noaa.gov/json/rtsw/rtsw_mag_1m.json"
SWPC_WIND_URL = "https://services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json"

#: The Kp-from-solar-wind fit.  Trained on OMNI2 2012-2021 (29,019 3-hour
#: intervals), tested on 2022-2024 (8,658): RMSE 0.73, correlation 0.84,
#: bias -0.16; for observed Kp >= 5 the RMSE is 1.43 and the bias -0.70.
#: The fit script is described in docs/spacewx.md.
KP_FIT = {"a": -1.0892145, "b": 2.7694599, "p": 0.6, "c": 35.518510}


#: A cached OMNI2 year file younger than this is not fetched again.
OMNI_REFETCH_SECONDS = 6 * 3600


class SolarWindError(ValueError):
    pass


@dataclass(frozen=True)
class SolarWind:
    """Hourly means, ascending, one row per hour (NaN where missing)."""
    hours: np.ndarray     # datetime64[h], the start of each hour
    by: np.ndarray        # nT, GSM
    bz: np.ndarray        # nT, GSM
    density: np.ndarray   # cm-3
    speed: np.ndarray     # km/s
    kp: np.ndarray        # observed Kp where the source carries it (OMNI), else NaN
    al: np.ndarray        # nT, where carried (OMNI), else NaN
    source: str
    sha256: str
    dst: np.ndarray | None = None    # nT, where carried (OMNI), else NaN
    pcn: np.ndarray | None = None    # northern polar cap index, where carried (OMNI), else NaN
    protons: np.ndarray | None = None   # (n, 6) integral proton flux >1..>60 MeV, cm-2 s-1 sr-1 (OMNI), else NaN

    def __post_init__(self):
        for name in ("dst", "pcn"):
            if getattr(self, name) is None:
                object.__setattr__(self, name, np.full(self.hours.size, np.nan))
        if self.protons is None:
            object.__setattr__(self, "protons", np.full((self.hours.size, 6), np.nan))

    def _index(self, when: datetime) -> int:
        i = int((np.datetime64(_naive_utc(when), "h") - self.hours[0]).astype(int))
        if not 0 <= i < self.hours.size:
            raise SolarWindError(
                f"{when}: outside the solar-wind record {self.hours[0]}..{self.hours[-1]} ({self.source})")
        return i

    def al_at(self, when: datetime) -> float:
        """The hour's observed AL (nT; NaN where the source has none)."""
        return float(self.al[self._index(when)])

    def dst_at(self, when: datetime) -> float:
        """The hour's Dst (nT; NaN where the source has none)."""
        return float(self.dst[self._index(when)])

    def pcn_at(self, when: datetime) -> float:
        """The hour's northern polar cap index (NaN where the source has none)."""
        return float(self.pcn[self._index(when)])

    def protons_at(self, when: datetime) -> np.ndarray:
        """The hour's integral proton fluxes >1, >2, >4, >10, >30, >60 MeV (NaN where missing)."""
        return self.protons[self._index(when)].copy()

    def hourly(self, when: datetime) -> dict:
        i = self._index(when)
        return {"by": float(self.by[i]), "bz": float(self.bz[i]),
                "density": float(self.density[i]), "speed": float(self.speed[i])}

    def kp_estimate(self, when: datetime) -> float:
        """The fitted Kp of the 3-hour UT interval containing ``when``."""
        start = _naive_utc(when).replace(minute=0, second=0, microsecond=0)
        start -= timedelta(hours=start.hour % 3)
        # The fit was made on these three hours: all must lie in the record
        # (fill values inside it are averaged over, as in the fit).
        sel = np.asarray([self._index(start + timedelta(hours=k)) for k in (-1, 0, 1)])
        x = np.nanmean(newell_coupling(self.speed[sel], self.by[sel], self.bz[sel])) if np.any(
            np.isfinite(self.speed[sel] + self.by[sel] + self.bz[sel])) else np.nan
        w = np.nanmean(np.sqrt(self.density[sel]) * self.speed[sel] ** 2) if np.any(
            np.isfinite(self.density[sel] + self.speed[sel])) else np.nan
        if not (np.isfinite(x) and np.isfinite(w)):
            raise SolarWindError(f"{when}: the solar wind is missing around this Kp interval ({self.source})")
        return kp_from_coupling(x, w)


def newell_coupling(speed, by, bz):
    """Newell et al. 2007 dPhi_MP/dt, in (km/s)^(4/3) nT^(2/3)."""
    speed, by, bz = (np.asarray(a, dtype=np.float64) for a in (speed, by, bz))
    bt = np.hypot(by, bz)
    theta = np.arctan2(by, bz)
    return speed ** (4.0 / 3.0) * bt ** (2.0 / 3.0) * np.abs(np.sin(theta / 2.0)) ** (8.0 / 3.0)


def boyle_potential(speed, b, bz):
    """Boyle et al. 1997 polar cap potential, kV (|B| and Bz GSM in nT, v in km/s)."""
    speed, b, bz = (np.asarray(a, dtype=np.float64) for a in (speed, b, bz))
    cos_t = np.clip(np.where(b > 0, bz / np.where(b > 0, b, 1.0), 1.0), -1.0, 1.0)
    return 1.0e-4 * speed ** 2 + 11.7 * b * np.sin(np.arccos(cos_t) / 2.0) ** 3


def kp_from_coupling(newell, viscous) -> float:
    """:data:`KP_FIT` applied to a mean Newell coupling and a mean sqrt(n) v^2."""
    f = KP_FIT
    kp = f["a"] + f["b"] * (1.0e-4 * float(newell)) ** f["p"] + f["c"] * 1.0e-7 * float(viscous)
    return float(min(9.0, max(0.0, kp)))


# ------------------------------------------------------------------ OMNI2

def _kp_from_code(code):
    code = np.asarray(code, dtype=np.int64)
    third = np.select([code % 10 == 3, code % 10 == 7], [1.0 / 3.0, 2.0 / 3.0], 0.0)
    return code // 10 + third


def parse_omni2(text: str, source: str = "", sha256: str = "") -> SolarWind:
    rows = []
    for line in text.splitlines():
        words = line.split()
        if not words:
            continue
        if len(words) < 55:
            raise SolarWindError(f"{source}: OMNI2 row with {len(words)} words, expected 55: {line[:40]!r}")
        rows.append([float(w) for w in words[:55]])
    if not rows:
        raise SolarWindError(f"{source}: no OMNI2 rows")
    d = np.asarray(rows)
    start = np.array([np.datetime64(f"{int(y):04d}-01-01T00", "h") for y in d[:, 0]])
    hours = start + ((d[:, 1] - 1) * 24 + d[:, 2]).astype("timedelta64[h]")
    order = np.argsort(hours)
    d, hours = d[order], hours[order]
    if np.any(np.diff(hours).astype(int) != 1):
        raise SolarWindError(f"{source}: OMNI2 rows are not consecutive hours")

    def word(n, fill):
        x = d[:, n - 1].copy()
        x[np.abs(x) >= fill] = np.nan
        return x

    kp = _kp_from_code(np.where(d[:, 38] >= 99, 0, d[:, 38]))
    kp[d[:, 38] >= 99] = np.nan
    return SolarWind(hours, word(16, 999.9), word(17, 999.9), word(24, 999.9), word(25, 9999.0),
                     kp, word(53, 99999.0), source, sha256, word(41, 99999.0), word(52, 999.9),
                     np.stack([word(43, 999999.0)] + [word(n, 99999.0) for n in range(44, 49)], axis=1))


def _concat(parts: list[SolarWind], source: str) -> SolarWind:
    parts = sorted(parts, key=lambda s: s.hours[0])
    for a, b in zip(parts, parts[1:]):
        if int((b.hours[0] - a.hours[-1]).astype(int)) != 1:
            raise SolarWindError(f"{source}: the files leave a gap between {a.hours[-1]} and {b.hours[0]}")
    sha = hashlib.sha256("".join(p.sha256 for p in parts).encode()).hexdigest()
    cat = lambda name: np.concatenate([getattr(p, name) for p in parts])
    return SolarWind(cat("hours"), cat("by"), cat("bz"), cat("density"), cat("speed"),
                     cat("kp"), cat("al"), source, sha, cat("dst"), cat("pcn"), cat("protons"))


def read_omni2(paths) -> SolarWind:
    paths = [Path(p) for p in ([paths] if isinstance(paths, (str, Path)) else paths)]
    parts = []
    for path in paths:
        data = path.read_bytes()
        parts.append(parse_omni2(data.decode("ascii", "replace"), str(path), hashlib.sha256(data).hexdigest()))
    return _concat(parts, ":".join(str(p) for p in paths))


def fetch_omni2(start: datetime, end: datetime, *, cache=None, opener=None) -> list[Path]:
    """The OMNI2 year files covering ``start``..``end``, cached; a year whose
    cached file ends before ``end`` is fetched again (the current year grows)."""
    root = Path(cache) if cache is not None else cache_root() / "omni"
    root.mkdir(parents=True, exist_ok=True)
    import urllib.request
    opener = opener or (lambda url: urllib.request.urlopen(url, timeout=120))
    out = []
    start, end = _naive_utc(start), _naive_utc(end)
    for year in range(start.year, end.year + 1):
        target = root / f"omni2_{year}.dat"
        if target.is_file():
            try:
                cached = read_omni2(target)
            except SolarWindError:
                cached = None                      # a damaged cache file is fetched again
            if cached is not None:
                last = np.datetime64(min(end, datetime(year, 12, 31, 23)), "h")
                # Covered: the file reaches the window's end with data in its
                # last day (a file padded with fill rows is not covered).
                tail = (cached.hours > last - np.timedelta64(24, "h")) & (cached.hours <= last)
                covered = cached.hours[-1] >= last and bool(np.any(np.isfinite(cached.speed[tail])))
                # OMNI2 lags by weeks: a file fetched within the last 6 hours
                # is as new as the server's, so it is not fetched again.
                fresh = time.time() - target.stat().st_mtime < OMNI_REFETCH_SECONDS
                if covered or fresh:
                    out.append(target)
                    continue
        part = unique_part(target)
        try:
            with opener(OMNI_URL.format(year=year)) as reply:
                part.write_bytes(reply.read())
            parse_omni2(part.read_bytes().decode("ascii", "replace"), str(part))
        except BaseException:
            part.unlink(missing_ok=True)
            raise
        part.replace(target)
        out.append(target)
    return out


# ------------------------------------------------------------------ SWPC real time

def parse_swpc(mag_text: str, wind_text: str, source: str = "swpc") -> SolarWind:
    """Hourly means of the active spacecraft's 1-minute RTSW records."""
    def records(text, fields):
        rows = json.loads(text)
        if not isinstance(rows, list):
            raise SolarWindError(f"{source}: RTSW JSON is not a list of records")
        out = {}
        for row in rows:
            if not isinstance(row, dict) or not row.get("active"):
                continue
            try:
                t = np.datetime64(str(row["time_tag"])[:16], "m")
            except (KeyError, ValueError):
                continue
            vals = [row.get(f) for f in fields]
            out[t] = [np.nan if v is None else float(v) for v in vals]
        return out

    mag = records(mag_text, ("by_gsm", "bz_gsm"))
    wind = records(wind_text, ("proton_density", "proton_speed"))
    if not mag or not wind:
        raise SolarWindError(f"{source}: no active-spacecraft records")
    first = min(min(mag), min(wind)).astype("datetime64[h]")
    last = max(max(mag), max(wind)).astype("datetime64[h]")
    hours = np.arange(first, last + np.timedelta64(1, "h"), np.timedelta64(1, "h"))
    acc = np.full((hours.size, 4), np.nan)

    def mean_into(table, cols):
        bins = {}
        for t, vals in table.items():
            bins.setdefault(int((t.astype("datetime64[h]") - first).astype(int)), []).append(vals)
        for i, vals in bins.items():
            arr = np.asarray(vals, dtype=np.float64)
            with np.errstate(invalid="ignore"):
                for j, col in enumerate(cols):
                    good = np.isfinite(arr[:, j])
                    if good.sum() >= 10:      # at least 10 of 60 minutes
                        acc[i, col] = arr[good, j].mean()

    mean_into(mag, (0, 1))
    mean_into(wind, (2, 3))
    nan = np.full(hours.size, np.nan)
    sha = hashlib.sha256((mag_text + wind_text).encode()).hexdigest()
    return SolarWind(hours, acc[:, 0], acc[:, 1], acc[:, 2], acc[:, 3], nan, nan.copy(), source, sha)


def fetch_swpc(*, opener=None) -> SolarWind:
    import urllib.request
    opener = opener or (lambda url: urllib.request.urlopen(url, timeout=120))
    with opener(SWPC_MAG_URL) as reply:
        mag = reply.read().decode("utf-8", "replace")
    with opener(SWPC_WIND_URL) as reply:
        wind = reply.read().decode("utf-8", "replace")
    return parse_swpc(mag, wind, "swpc")


def load(source: str, start: datetime, end: datetime) -> SolarWind:
    """``"omni"`` (fetched), ``"swpc"`` (the last 24 hours, fetched) or OMNI2 path(s) joined by ``:``."""
    text = str(source or "").strip()
    if text == "omni":
        return read_omni2(fetch_omni2(start, end))
    if text == "swpc":
        return fetch_swpc()
    if not text:
        raise SolarWindError("no solar-wind source: 'omni', 'swpc' or OMNI2 file path(s)")
    return read_omni2(text.split(":"))


__all__ = ["SolarWind", "SolarWindError", "KP_FIT", "newell_coupling", "boyle_potential",
           "kp_from_coupling", "parse_omni2", "read_omni2", "fetch_omni2", "parse_swpc",
           "fetch_swpc", "load"]
