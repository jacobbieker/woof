"""IGRF-14 main-field synthesis on the host: inclination, modip, dip latitude.

The ionospheric maps (CCIR/URSI, Jones & Gallet 1965) are expanded in Rawer's
modified dip latitude ``modip``, ``tan(modip) = I / sqrt(cos(lat))`` with the
inclination ``I`` in radians taken at 300 km (Rawer 1963; Bilitza et al. 2022,
Rev. Geophys. 60, e2022RG000792, sec. 3.1).  The F1 layer uses the magnetic dip
latitude ``atan(tan(I) / 2)``.

The field is the IAGA International Geomagnetic Reference Field, 14th
generation (Alken et al. 2025, Earth Planets Space; coefficients published by
NOAA NCEI for IAGA V-MOD), evaluated here from its Schmidt semi-normalised
Gauss coefficients by the standard spherical-harmonic sums (Langel 1987, Main
field, in Geomagnetism vol. 1; Winch et al. 2005, Geophys. J. Int. 160, 487).
Between the 5-year epochs the coefficients are interpolated linearly; after
the last definitive epoch the secular-variation column extrapolates, and a
date beyond its 5-year window is refused.

The coefficient file is fetched once into the space-weather cache and checked
against a pinned size and SHA-256; it is data, never shipped in the wheel.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
from pathlib import Path

import numpy as np

from woof.spacewx.assets import PinnedAsset, fetch_pinned
from woof.spacewx.solar_irradiance import naive_utc as _naive_utc

#: IAGA IGRF-14 coefficients (NOAA NCEI mirror of the IAGA V-MOD release).
IGRF14 = PinnedAsset(
    name="igrf14coeffs.txt",
    url="https://www.ngdc.noaa.gov/IAGA/vmod/coeffs/igrf14coeffs.txt",
    bytes=42411,
    sha256="8f8d88403028fc4ee92c4f38d97b46e0a87e2cfc496045b43c9e26c1d6b0903c",
    subdir="igrf",
)
#: The IGRF reference radius, km.
REFERENCE_RADIUS_KM = 6371.2
#: WGS84 semi-major axis (km) and flattening.
WGS84_A_KM = 6378.137
WGS84_F = 1.0 / 298.257223563
#: The height of the inclination the CCIR/URSI maps use, km.
MODIP_HEIGHT_KM = 300.0


class IgrfError(ValueError):
    pass


@dataclass(frozen=True)
class IgrfModel:
    """Gauss coefficients ``g[n, m]``, ``h[n, m]`` at each epoch plus the SV."""

    epochs: np.ndarray        # (nepoch,) decimal years
    g: np.ndarray             # (nepoch, nmax + 1, nmax + 1) nT
    h: np.ndarray
    sv_g: np.ndarray          # (nmax + 1, nmax + 1) nT / yr after the last epoch
    sv_h: np.ndarray
    sha256: str

    @property
    def nmax(self) -> int:
        return self.g.shape[1] - 1

    def coefficients(self, year: float) -> tuple[np.ndarray, np.ndarray]:
        """``(g, h)`` at decimal ``year``."""
        year = float(year)
        first, last = float(self.epochs[0]), float(self.epochs[-1])
        if year < first or year > last + 5.0:
            raise IgrfError(f"IGRF-14 covers {first:.0f}-{last + 5.0:.0f}; got {year:.3f}")
        if year >= last:
            dt = year - last
            return self.g[-1] + dt * self.sv_g, self.h[-1] + dt * self.sv_h
        i = int(np.searchsorted(self.epochs, year, side="right")) - 1
        w = (year - self.epochs[i]) / (self.epochs[i + 1] - self.epochs[i])
        return ((1.0 - w) * self.g[i] + w * self.g[i + 1],
                (1.0 - w) * self.h[i] + w * self.h[i + 1])


def parse(text: str, sha256: str = "") -> IgrfModel:
    """The IAGA ``igrfNNcoeffs.txt`` table: ``g/h n m <epochs...> <SV>``."""
    rows, epochs = [], None
    for line in text.splitlines():
        words = line.split()
        if not words:
            continue
        if words[0] == "g/h":
            epochs = [float(w) for w in words[3:-1]]
            continue
        if words[0] not in ("g", "h") or epochs is None:
            continue
        values = [float(w) for w in words[3:]]
        if len(values) != len(epochs) + 1:
            raise IgrfError(f"IGRF row {' '.join(words[:3])}: {len(values)} values for {len(epochs)} epochs + SV")
        rows.append((words[0], int(words[1]), int(words[2]), values))
    if epochs is None or not rows:
        raise IgrfError("not an IAGA IGRF coefficient table (no 'g/h n m' header)")
    nmax = max(n for _, n, _, _ in rows)
    ne = len(epochs)
    g = np.zeros((ne, nmax + 1, nmax + 1))
    h = np.zeros_like(g)
    sv_g = np.zeros((nmax + 1, nmax + 1))
    sv_h = np.zeros_like(sv_g)
    for kind, n, m, values in rows:
        if not 0 <= m <= n:
            raise IgrfError(f"IGRF row {kind} {n} {m}: order outside 0..n")
        (g if kind == "g" else h)[:, n, m] = values[:-1]
        (sv_g if kind == "g" else sv_h)[n, m] = values[-1]
    return IgrfModel(np.asarray(epochs), g, h, sv_g, sv_h, sha256)


def load(path=None) -> IgrfModel:
    """The pinned IGRF-14 table (fetched once), or the file ``path``."""
    target = Path(path) if path is not None else fetch_pinned(IGRF14)
    data = target.read_bytes()
    return parse(data.decode("ascii"), hashlib.sha256(data).hexdigest())


def decimal_year(when: datetime) -> float:
    when = _naive_utc(when)
    start = datetime(when.year, 1, 1)
    length = (datetime(when.year + 1, 1, 1) - start).total_seconds()
    return when.year + (when - start).total_seconds() / length


def _geodetic_to_geocentric(lat_deg, alt_km):
    """Geocentric radius (km), colatitude (rad) and the rotation angle
    (geodetic minus geocentric latitude, rad) of WGS84 geodetic points."""
    lat = np.deg2rad(lat_deg)
    e2 = WGS84_F * (2.0 - WGS84_F)
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    n = WGS84_A_KM / np.sqrt(1.0 - e2 * sin_lat**2)
    rho = (n + alt_km) * cos_lat
    z = (n * (1.0 - e2) + alt_km) * sin_lat
    r = np.hypot(rho, z)
    lat_gc = np.arctan2(z, rho)
    return r, 0.5 * np.pi - lat_gc, lat - lat_gc


def field(model: IgrfModel, year: float, lat_deg, lon_deg, alt_km):
    """Geodetic north, east and down components (nT) at WGS84 points."""
    lat_deg, lon_deg = np.broadcast_arrays(np.asarray(lat_deg, float), np.asarray(lon_deg, float))
    shape = lat_deg.shape
    lat_deg, lon_deg = lat_deg.ravel(), lon_deg.ravel()
    alt = np.broadcast_to(np.asarray(alt_km, float), shape).ravel()
    g, h = model.coefficients(year)
    nmax = model.nmax
    r, theta, delta = _geodetic_to_geocentric(lat_deg, alt)
    # Keep sin(theta) off zero at the poles; the east component there is the
    # limit of its finite neighbours to well below a nanotesla.
    st = np.maximum(np.sin(theta), 1.0e-10)
    ct = np.cos(theta)
    phi = np.deg2rad(lon_deg)
    npts = lat_deg.size
    # Schmidt semi-normalised associated Legendre functions and d/dtheta.
    p = np.zeros((nmax + 1, nmax + 1, npts))
    dp = np.zeros_like(p)
    p[0, 0] = 1.0
    for n in range(1, nmax + 1):
        if n == 1:
            p[1, 1] = st
            dp[1, 1] = ct
        else:
            k = np.sqrt((2.0 * n - 1.0) / (2.0 * n))
            p[n, n] = k * st * p[n - 1, n - 1]
            dp[n, n] = k * (ct * p[n - 1, n - 1] + st * dp[n - 1, n - 1])
        for m in range(n):
            a = np.sqrt(float(n * n - m * m))
            b = np.sqrt(float((n - 1) ** 2 - m * m)) if n - 1 >= m else 0.0
            prev2 = p[n - 2, m] if n >= 2 else 0.0
            dprev2 = dp[n - 2, m] if n >= 2 else 0.0
            p[n, m] = ((2.0 * n - 1.0) * ct * p[n - 1, m] - b * prev2) / a
            dp[n, m] = ((2.0 * n - 1.0) * (ct * dp[n - 1, m] - st * p[n - 1, m]) - b * dprev2) / a
    br = np.zeros(npts)
    bt = np.zeros(npts)
    bp = np.zeros(npts)
    ratio = REFERENCE_RADIUS_KM / r
    for n in range(1, nmax + 1):
        rn = ratio ** (n + 2)
        for m in range(n + 1):
            cm, sm = np.cos(m * phi), np.sin(m * phi)
            gh = g[n, m] * cm + h[n, m] * sm
            br += (n + 1) * rn * gh * p[n, m]
            bt -= rn * gh * dp[n, m]
            bp += rn * m * (g[n, m] * sm - h[n, m] * cm) * p[n, m] / st
    # Geocentric north/east/down, then rotate into the geodetic frame.
    x_gc, y, z_gc = -bt, bp, -br
    cd, sd = np.cos(delta), np.sin(delta)
    x = x_gc * cd + z_gc * sd
    z = z_gc * cd - x_gc * sd
    return x.reshape(shape), y.reshape(shape), z.reshape(shape)


def inclination(model: IgrfModel, year: float, lat_deg, lon_deg, alt_km=MODIP_HEIGHT_KM):
    """Dip angle (deg, positive downward)."""
    x, y, z = field(model, year, lat_deg, lon_deg, alt_km)
    return np.rad2deg(np.arctan2(z, np.hypot(x, y)))


def modip(inc_deg, lat_deg):
    """Rawer's modified dip latitude (deg): ``atan(I_rad / sqrt(cos(lat)))``."""
    coslat = np.maximum(np.cos(np.deg2rad(np.asarray(lat_deg, float))), 1.0e-12)
    return np.rad2deg(np.arctan(np.deg2rad(np.asarray(inc_deg, float)) / np.sqrt(coslat)))


def dip_latitude(inc_deg):
    """Magnetic dip latitude (deg): ``atan(tan(I) / 2)``."""
    return np.rad2deg(np.arctan(0.5 * np.tan(np.deg2rad(np.asarray(inc_deg, float)))))


def magnetic_coordinates(lat_deg, lon_deg, when: datetime, model: IgrfModel | None = None):
    """``(inclination, modip, dip latitude)`` in degrees at 300 km for ``when``."""
    model = model or load()
    inc = inclination(model, decimal_year(when), lat_deg, lon_deg)
    return inc, modip(inc, lat_deg), dip_latitude(inc)
