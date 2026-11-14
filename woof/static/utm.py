"""WGS84 Universal Transverse Mercator, forward and inverse, in numpy.

GEM-15 publishes each 1x1-degree cell in its own UTM zone (EPSG 326zz
north, 327zz south).  The fetch needs the footprint in a cell's plane to
choose which 512 x 512 blocks to read; the mosaic (Rust ``static-fields``,
``raster/proj.rs::Utm``) needs the same transform to place each block on
the geographic lattice.  Both use the Krueger series to sixth order in the
third flattening (Karney 2011, "Transverse Mercator with an accuracy of a
few nanometers", J. Geodesy 85, 475), which agrees with PROJ's ``tmerc``
to well under a millimetre inside a zone and its neighbours; the tests
hold it to pyproj.
"""
from __future__ import annotations

import numpy as np

#: WGS84.
A = 6_378_137.0
F = 1.0 / 298.257223563
K0 = 0.9996
FALSE_EASTING = 500_000.0
FALSE_NORTHING_SOUTH = 10_000_000.0

_N = F / (2.0 - F)
_E = np.sqrt(F * (2.0 - F))
_RECT = A / (1.0 + _N) * (1.0 + _N**2 / 4.0 + _N**4 / 64.0 + _N**6 / 256.0)
_n = _N
_ALPHA = (
    _n / 2 - 2 * _n**2 / 3 + 5 * _n**3 / 16 + 41 * _n**4 / 180
    - 127 * _n**5 / 288 + 7891 * _n**6 / 37800,
    13 * _n**2 / 48 - 3 * _n**3 / 5 + 557 * _n**4 / 1440
    + 281 * _n**5 / 630 - 1983433 * _n**6 / 1935360,
    61 * _n**3 / 240 - 103 * _n**4 / 140 + 15061 * _n**5 / 26880
    + 167603 * _n**6 / 181440,
    49561 * _n**4 / 161280 - 179 * _n**5 / 168
    + 6601661 * _n**6 / 7257600,
    34729 * _n**5 / 80640 - 3418889 * _n**6 / 1995840,
    212378941 * _n**6 / 319334400,
)
_BETA = (
    _n / 2 - 2 * _n**2 / 3 + 37 * _n**3 / 96 - _n**4 / 360
    - 81 * _n**5 / 512 + 96199 * _n**6 / 604800,
    _n**2 / 48 + _n**3 / 15 - 437 * _n**4 / 1440 + 46 * _n**5 / 105
    - 1118711 * _n**6 / 3870720,
    17 * _n**3 / 480 - 37 * _n**4 / 840 - 209 * _n**5 / 4480
    + 5569 * _n**6 / 90720,
    4397 * _n**4 / 161280 - 11 * _n**5 / 504
    - 830251 * _n**6 / 7257600,
    4583 * _n**5 / 161280 - 108847 * _n**6 / 3991680,
    20648693 * _n**6 / 638668800,
)


def zone_of_epsg(epsg: int) -> tuple[int, bool]:
    """``(zone, north)`` of a WGS84 UTM EPSG code; refuses anything else."""
    epsg = int(epsg)
    if 32601 <= epsg <= 32660:
        return epsg - 32600, True
    if 32701 <= epsg <= 32760:
        return epsg - 32700, False
    raise ValueError(f"EPSG:{epsg} is not a WGS84 UTM zone (32601-32660, "
                     "32701-32760)")


def central_meridian(zone: int) -> float:
    return -183.0 + 6.0 * int(zone)


def forward(epsg: int, lon, lat):
    """Longitude/latitude (degrees) -> UTM easting/northing (metres)."""
    zone, north = zone_of_epsg(epsg)
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    dlon = np.radians((lon - central_meridian(zone) + 180.0) % 360.0 - 180.0)
    phi = np.radians(lat)
    sin_phi = np.sin(phi)
    tau = np.sinh(np.arctanh(sin_phi) - _E * np.arctanh(_E * sin_phi))
    xi_p = np.arctan2(tau, np.cos(dlon))
    eta_p = np.arctanh(np.sin(dlon) / np.sqrt(1.0 + tau * tau))
    xi, eta = xi_p.copy(), eta_p.copy()
    for j, alpha in enumerate(_ALPHA, start=1):
        xi += alpha * np.sin(2 * j * xi_p) * np.cosh(2 * j * eta_p)
        eta += alpha * np.cos(2 * j * xi_p) * np.sinh(2 * j * eta_p)
    easting = FALSE_EASTING + K0 * _RECT * eta
    northing = K0 * _RECT * xi + (0.0 if north else FALSE_NORTHING_SOUTH)
    return easting, northing


def inverse(epsg: int, easting, northing):
    """UTM easting/northing (metres) -> longitude/latitude (degrees)."""
    zone, north = zone_of_epsg(epsg)
    x = np.asarray(easting, dtype=np.float64)
    y = np.asarray(northing, dtype=np.float64)
    xi = (y - (0.0 if north else FALSE_NORTHING_SOUTH)) / (K0 * _RECT)
    eta = (x - FALSE_EASTING) / (K0 * _RECT)
    xi_p, eta_p = xi.copy(), eta.copy()
    for j, beta in enumerate(_BETA, start=1):
        xi_p -= beta * np.sin(2 * j * xi) * np.cosh(2 * j * eta)
        eta_p -= beta * np.cos(2 * j * xi) * np.sinh(2 * j * eta)
    tau_p = np.sin(xi_p) / np.sqrt(np.sinh(eta_p) ** 2 + np.cos(xi_p) ** 2)
    tau = tau_p.copy()
    for _ in range(6):  # Newton; converges to round-off in 2-3 steps
        sigma = np.sinh(_E * np.arctanh(_E * tau / np.sqrt(1.0 + tau * tau)))
        tau_i = tau * np.sqrt(1.0 + sigma * sigma) - sigma * np.sqrt(
            1.0 + tau * tau)
        tau = tau + (tau_p - tau_i) / np.sqrt(1.0 + tau_i * tau_i) * (
            (1.0 + (1.0 - _E * _E) * tau * tau)
            / ((1.0 - _E * _E) * np.sqrt(1.0 + tau * tau)))
    lat = np.degrees(np.arctan(tau))
    lon = central_meridian(zone) + np.degrees(np.arctan2(np.sinh(eta_p),
                                                         np.cos(xi_p)))
    lon = (lon + 180.0) % 360.0 - 180.0
    return lon, lat


__all__ = ["central_meridian", "forward", "inverse", "zone_of_epsg"]
