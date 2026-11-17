"""Geomagnetic and solar activity indices: Kp, ap, Ap and F10.7.

The source is GFZ Potsdam's ``Kp_ap_Ap_SN_F107_since_1932.txt`` (Matzka et al.
2021, Space Weather 19, e2020SW002641; CC BY 4.0).  It carries the 3-hourly Kp
and ap, the daily Ap and the observed and adjusted 10.7 cm solar radio flux
(Tapping 2013, Space Weather 11, 394), up to the current day as nowcast
values.  The file's sunspot-number column is licensed CC BY-NC and is never
read.  A local file in the same format can stand in for the download.

The ionosphere uses the observed flux F10.7obs, as GFZ and Tapping recommend
for ionospheric work.  Two drivers are offered, both available at forecast
time:

* ``"daily"``: the previous UT day's F10.7obs.
* ``"81day"``: the mean F10.7obs of the 81 UT days ending on the previous
  day (a trailing form of the centred 81-day mean F10.7A).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import hashlib
from pathlib import Path

import numpy as np

from woof.spacewx.solar_irradiance import cache_root

GFZ_URL = "https://kp.gfz.de/app/files/Kp_ap_Ap_SN_F107_since_1932.txt"
GFZ_NAME = "Kp_ap_Ap_SN_F107_since_1932.txt"
F107_MODES = ("daily", "81day")


class IndicesError(ValueError):
    pass


@dataclass(frozen=True)
class GeoIndices:
    days: np.ndarray        # (n,) datetime64[D], consecutive UT days
    kp: np.ndarray          # (n, 8) 3-hourly Kp, NaN where missing
    ap: np.ndarray          # (n, 8) 3-hourly ap
    ap_daily: np.ndarray    # (n,) Ap
    f107: np.ndarray        # (n,) F10.7obs, sfu
    source: str
    sha256: str

    def _row(self, day: date) -> int:
        i = int((np.datetime64(day, "D") - self.days[0]).astype(int))
        if not 0 <= i < self.days.size:
            raise IndicesError(f"{day}: outside the index record {self.days[0]}..{self.days[-1]} ({self.source})")
        return i

    def f107_driver(self, when: datetime, mode: str = "81day") -> float:
        """The F10.7 that drives the ionosphere at ``when`` (see module doc)."""
        if mode not in F107_MODES:
            raise IndicesError(f"F10.7 driver {mode!r}: one of {F107_MODES}")
        last = self._row((when - timedelta(days=1)).date())
        first = last if mode == "daily" else last - 80
        if first < 0:
            raise IndicesError(f"{when}: the record starts {self.days[0]}, too late for an {mode} F10.7")
        values = self.f107[first:last + 1]
        values = values[np.isfinite(values)]
        if values.size == 0 or (mode == "81day" and values.size < 60):
            raise IndicesError(f"{when}: too few observed F10.7 values for the {mode} driver ({self.source})")
        return float(values.mean())

    def has_kp(self, when: datetime) -> bool:
        """Whether the record holds an observed Kp for ``when``'s interval."""
        try:
            return bool(np.isfinite(self.kp[self._row(when.date()), when.hour // 3]))
        except IndicesError:
            return False

    def kp_at(self, when: datetime) -> float:
        """The 3-hourly Kp of the interval containing ``when``."""
        i = self._row(when.date())
        value = self.kp[i, when.hour // 3]
        if not np.isfinite(value):
            raise IndicesError(f"{when}: Kp missing in {self.source}")
        return float(value)


def parse(text: str, source: str = "", sha256: str = "") -> GeoIndices:
    days, kp, ap, ap_daily, f107 = [], [], [], [], []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        words = line.split()
        if len(words) < 28:
            raise IndicesError(f"{source}: index row with {len(words)} columns, expected 28: {line[:40]!r}")
        days.append(np.datetime64(f"{int(words[0]):04d}-{int(words[1]):02d}-{int(words[2]):02d}", "D"))
        kp.append([float(w) for w in words[7:15]])
        ap.append([float(w) for w in words[15:23]])
        ap_daily.append(float(words[23]))
        # words[24] is the sunspot number (CC BY-NC): deliberately not read.
        f107.append(float(words[25]))
    if not days:
        raise IndicesError(f"{source}: no index rows")
    days = np.asarray(days)
    if np.any(np.diff(days).astype(int) != 1):
        raise IndicesError(f"{source}: index rows are not consecutive UT days")
    kp = np.asarray(kp)
    ap = np.asarray(ap)
    ap_daily = np.asarray(ap_daily)
    f107 = np.asarray(f107)
    kp[kp < 0] = np.nan
    ap[ap < 0] = np.nan
    ap_daily[ap_daily < 0] = np.nan
    f107[f107 <= 0] = np.nan
    return GeoIndices(days, kp, ap, ap_daily, f107, source, sha256)


def read(path) -> GeoIndices:
    data = Path(path).read_bytes()
    return parse(data.decode("ascii", "replace"), str(path), hashlib.sha256(data).hexdigest())


def fetch_gfz(needed: datetime | None = None, *, cache=None, opener=None, kp: bool = False) -> Path:
    """GFZ's index file in the cache, fetched again when it ends before ``needed``
    (with ``kp``: when it lacks the Kp of ``needed``'s 3-hour interval, which GFZ
    adds through the day)."""
    root = Path(cache) if cache is not None else cache_root() / "indices"
    root.mkdir(parents=True, exist_ok=True)
    target = root / GFZ_NAME
    if target.is_file() and needed is not None:
        try:
            current = read(target)
            if current.days[-1] >= np.datetime64(needed.date(), "D") and (
                    not kp or current.has_kp(needed)):
                return target
        except IndicesError:
            pass
    elif target.is_file():
        return target
    import urllib.request
    opener = opener or (lambda url: urllib.request.urlopen(url, timeout=120))
    part = target.with_name(target.name + ".part")
    with opener(GFZ_URL) as reply:
        part.write_bytes(reply.read())
    parse(part.read_bytes().decode("ascii", "replace"), str(part))
    part.replace(target)
    return target


def load(source: str = "gfz", needed: datetime | None = None, *, kp: bool = False) -> GeoIndices:
    """``"gfz"`` (fetched) or a path to a file in GFZ's format."""
    text = str(source or "").strip()
    if text in ("", "gfz"):
        return read(fetch_gfz(needed, kp=kp))
    return read(text)
