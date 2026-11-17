"""Climatological ionosphere on the GPU: foF2, hmF2, foE, F1 and vertical TEC.

Layer L3a of the space-weather design (docs/spacewx.md).  An IRI-type electron
density profile is built on every requested column from published formulas:
CCIR/URSI numerical maps for foF2 and M(3000)F2 (Jones & Gallet 1965; CCIR
Report 340; URSI 1988), the IRI E, F1, hmF2 and thickness relations, and
Epstein layers.  The formulas and their references are listed at the top of
``woof/core/kernels/spacewx_ionosphere.cu``; PyIRI (Forsythe et al. 2023, Space
Weather 21, e2022SW003352; MIT licence) was read for the arrangement of the
map expansions, and its coefficient files are fetched (pinned by SHA-256) as
data.  No IRI or PyIRI code is run or copied.

Space weather enters through the solar activity index: the F10.7 radio flux
drives the maps (as IG12 and R12) and foE.  The maps are monthly medians, so
the profile is a climatology for that activity; it does not respond to a
geomagnetic storm (the storm-time layers are later work).

The host does the part that is the same for every column: it contracts the
map coefficients with the diurnal Fourier series of the UT, blends the two
nearest monthly maps (mid-month to mid-month, linear in time) and the two
activity levels (foF2 in IG12, M(3000)F2 in R12, linear between 0 and 100).
The device evaluates the geographic functions and the profile per column.
"""
from __future__ import annotations

from datetime import datetime
import hashlib

import numpy as np

from woof.spacewx import igrf
from woof.spacewx.assets import PinnedAsset, fetch_pinned

PYIRI_COMMIT = "c0a50e0af6370f00dd24c1f8900442fe818a331b"
_PYIRI_RAW = f"https://raw.githubusercontent.com/victoriyaforsythe/PyIRI/{PYIRI_COMMIT}/PyIRI/coefficients"
_CCIR_SHA256 = (
    "56646023cabd38a6cebe94f529d3cabba15f22cae02ec0ee84758b26938fe87e",
    "a2e9b7a462c30830251d94de4b3e18f761ac655e019a07ac286e88e36ad92761",
    "ab849d91a7801791dd5d9866512b5a257a713650bd940a761cbdddd90b876ddb",
    "8ee15c4f27544c041a70d33c89fa117874a815b4848ffd0dbc344813aeaef37e",
    "a169404dd1f1d9ec14e24cccd3d1d20729412f04fb935d16612800b35bc8075c",
    "361660d44a8503c9e9004a783fc7a6e61fad24e5fb8fdc00b93fc5f6ce720b66",
    "c80d0fd0e9eb53dc65e3db96231c9cbac23f7471e9f864bee878f3000c49690d",
    "7e07ae4a6e05e9eea0b3d8a4e8ecf590cdb1c86db4d79ab4792afd929761a989",
    "8c34980f9933c846f3449d5df2543e0c808929f321c01164d82b4c249b4c002c",
    "d1211c10e5d3e6e12dd6cbfef20711d171577df03d5424dbd60c1ac6a34a8be8",
    "8195a69009eb02806f22cd3614cf44e01f60177fae335450de12c717f1af96a4",
    "f16f33af514852942cf461d44446773854342bc806de428289fb0165c50db3ac",
)
_URSI_SHA256 = (
    "d5fc350415f79036118fbf96c538a66a2c0725922ceb6409603b991db13dc2f5",
    "4acfeddc42a440e74b3277c347a6611e959523e420a8c6e14b0384a4fd6f8314",
    "d1d122a20e17c0fcd95adea0efdd656c5185fc229c252408f6ea600dc6d6a74e",
    "cf8c777ede1f35afaa29ada8f1cab12c7b3e5cd01bc9662a7ae5a72aec7b32f1",
    "16904f87010e81eca4265b0caecc61bd49b54bc218fb663652e76d84c9ebbf8a",
    "1c8173174a5167dc49d9ad639d97ad5fbe5d8e2a19fcce7dd367778a1734d0a7",
    "64c38500668c31c63cd548706b0e97017b78e6473435fb0ffccc32b6e6e223f0",
    "4d3b81b25a4221d3535aee9a33ae588f9b51c2a4482c3c9f55588860e14e12e8",
    "7bb1884406fd93ee55cb3bcf216af50b04a694ab4b10903400e273dada62fdb8",
    "851502b6c24828b2ecf50efe9d4e7bf7239f4a131bfb03619c993a962cfccf67",
    "dae9d94ce7b434c088fb8d5f4ccd1d2d3f797cc7c5cca8a699ceb6b0cc17bb9e",
    "86085d89cccb8f7b8f9318925c689bafb238aa00a3d816f6ff5d94b29d63a151",
)
FAMILIES = ("ccir", "ursi")

# Expansion sizes (Jones & Graham 1966): diurnal order nj, geographic terms nk
# and the highest sine power per longitude order Q.
NJ_F2, NK_F2, Q_F2 = 13, 76, (12, 12, 9, 5, 2, 1, 1, 1, 1)
NJ_M3, NK_M3, Q_M3 = 9, 49, (7, 8, 6, 3, 2, 1, 1)
NCOEF = NK_F2 + NK_M3
#: Default TEC integration: 60-2000 km in 1 km cells (no plasmasphere).
ALT_LO_KM, ALT_HI_KM, NALT = 60.0, 2000.0, 1940
OUTPUTS = ("fof2", "m3000", "hmf2", "nmf2", "foe", "fof1", "hmf1", "b_bot", "b_top", "tec")


class IonosphereError(ValueError):
    pass


def map_asset(family: str, month: int) -> PinnedAsset:
    if family not in FAMILIES:
        raise IonosphereError(f"map family {family!r}: one of {FAMILIES}")
    if not 1 <= int(month) <= 12:
        raise IonosphereError(f"month {month}: 1-12")
    name = f"{family}{int(month) + 10}.asc"
    shas = _CCIR_SHA256 if family == "ccir" else _URSI_SHA256
    return PinnedAsset(name=name, url=f"{_PYIRI_RAW}/{family.upper()}/{name}",
                       bytes=44300 if family == "ccir" else 30628,
                       sha256=shas[int(month) - 1], subdir="ccir")


def parse_asc(text: str) -> np.ndarray:
    """Values of a Fortran ``(1X,4E15.8)`` file, in order."""
    values = []
    for line in text.splitlines():
        body = line[1:]
        for i in range(0, len(body.rstrip()), 15):
            field = body[i:i + 15].strip()
            if field:
                values.append(float(field))
    return np.asarray(values)


def _reshape(values, nj, nk):
    size = nj * nk * 2
    if values.size < size:
        raise IonosphereError(f"coefficient file holds {values.size} values, need {size}")
    return np.reshape(values[:size], (nj, nk, 2), order="F")


def read_maps(month: int, family: str = "ccir", *, cache=None, loader=None):
    """``(U_foF2 (13, 76, 2), U_M3000 (9, 49, 2))`` for ``month``; the last axis
    is the activity level (index 0, then 100).  M(3000)F2 is always CCIR's."""
    loader = loader or (lambda asset: fetch_pinned(asset, cache=cache).read_text("ascii"))
    ccir = parse_asc(loader(map_asset("ccir", month)))
    if ccir.size != NJ_F2 * NK_F2 * 2 + NJ_M3 * NK_M3 * 2:
        raise IonosphereError(f"CCIR month {month}: {ccir.size} values, expected 2858")
    m3 = _reshape(ccir[NJ_F2 * NK_F2 * 2:], NJ_M3, NK_M3)
    f2 = _reshape(ccir, NJ_F2, NK_F2)
    if family == "ursi":
        f2 = _reshape(parse_asc(loader(map_asset("ursi", month))), NJ_F2, NK_F2)
    return f2, m3


def diurnal(nj: int, ut_hours: float) -> np.ndarray:
    """``[1, sin h, cos h, sin 2h, cos 2h, ...]`` (nj terms), h = 15 UT - 180 deg."""
    hou = np.deg2rad(15.0 * float(ut_hours) - 180.0)
    out = np.empty(nj)
    out[0] = 1.0
    for idx in range(1, nj):
        order = (idx + 1) // 2
        out[idx] = np.sin(order * hou) if idx % 2 else np.cos(order * hou)
    return out


def geographic(q, lat_deg, lon_deg, modip_deg) -> np.ndarray:
    """The geographic functions G_k (nk, ncol) of one expansion."""
    lat, lon = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    smu, clat = np.sin(np.deg2rad(modip_deg)), np.cos(lat)
    rows = []
    for j, qj in enumerate(q):
        for i in range(qj):
            base = smu**i * clat**j
            if j == 0:
                rows.append(base)
            else:
                rows.append(base * np.cos(j * lon))
                rows.append(base * np.sin(j * lon))
    return np.asarray(rows)


def f107_to_r12(f107: float) -> float:
    """CCIR's F10.7 = 63.75 + 0.728 R12 + 8.9e-4 R12^2, inverted (R12 >= 0)."""
    f107 = max(float(f107), 63.75)
    a, b, c = 8.9e-4, 0.728, 63.75 - f107
    return (-b + np.sqrt(b * b - 4.0 * a * c)) / (2.0 * a)


def r12_to_ig12(r12: float) -> float:
    """IG12 from R12 (sunspot number v2; Forsythe et al. 2023 fit)."""
    return -11.5634 + 1.5332 * r12 - 0.0031 * r12 * r12


def season(month: int) -> int:
    """The foE season index: -1 Nov-Feb, 0 Mar/Apr/Sep/Oct, +1 May-Aug."""
    if month in (11, 12, 1, 2):
        return -1
    if month in (5, 6, 7, 8):
        return 1
    return 0


def _mid_month(year: int, month: int) -> datetime:
    while month < 1:
        month += 12
        year -= 1
    while month > 12:
        month -= 12
        year += 1
    return datetime(year, month, 15)


def month_weights(when: datetime) -> tuple[tuple[int, float], tuple[int, float]]:
    """The two monthly maps around ``when`` and their linear weights in time."""
    when = when.replace(tzinfo=None)
    here = _mid_month(when.year, when.month)
    if when >= here:
        before, after = here, _mid_month(when.year, when.month + 1)
    else:
        before, after = _mid_month(when.year, when.month - 1), here
    w_after = (when - before).total_seconds() / (after - before).total_seconds()
    return (before.month, 1.0 - w_after), (after.month, w_after)


def subsolar_point(when: datetime) -> tuple[float, float]:
    """Subsolar latitude and longitude (deg), low-precision solar coordinates
    (Astronomical Almanac, sec. C; good to about 0.01 deg)."""
    when = when.replace(tzinfo=None)
    j2000 = datetime(2000, 1, 1, 12)
    t = (when - j2000).total_seconds() / 86400.0 / 36525.0
    mean_lon = np.mod(280.460 + 36000.771 * t, 360.0)
    anomaly = np.deg2rad(np.mod(357.527723 + 35999.05034 * t, 360.0))
    ecl_lon = np.deg2rad(mean_lon + 1.914666471 * np.sin(anomaly) + 0.01999464 * np.sin(2.0 * anomaly))
    eps = np.deg2rad(23.439291 - 0.0130042 * t)
    dec = np.arcsin(np.sin(eps) * np.sin(ecl_lon))
    ra = np.rad2deg(np.arctan2(np.cos(eps) * np.sin(ecl_lon), np.cos(ecl_lon)))
    gmst_s = 67310.54841 + (876600.0 * 3600.0 + 8640184.812866) * t + 0.093104 * t**2 - 6.2e-6 * t**3
    gmst = np.mod(gmst_s, 86400.0) / 240.0
    lon = np.mod(ra - gmst + 180.0, 360.0) - 180.0
    return float(np.rad2deg(dec)), float(lon)


class Ionosphere:
    """The L3a ionosphere on a fixed set of columns (lat/lon in degrees)."""

    def __init__(self, lat_deg, lon_deg, *, when: datetime, family: str = "ccir",
                 alt_lo=ALT_LO_KM, alt_hi=ALT_HI_KM, nalt=NALT, igrf_model=None, map_loader=None):
        if family not in FAMILIES:
            raise IonosphereError(f"map family {family!r}: one of {FAMILIES}")
        if not (0.0 <= alt_lo < alt_hi) or int(nalt) < 1:
            raise IonosphereError(f"TEC range {alt_lo}-{alt_hi} km in {nalt} cells is empty")
        lat = np.asarray(lat_deg, dtype=np.float64).ravel()
        lon = np.asarray(lon_deg, dtype=np.float64).ravel()
        if lat.shape != lon.shape or lat.size == 0:
            raise IonosphereError("lat and lon must be non-empty and of one size")
        if np.any(np.abs(lat) > 90.0) or not np.all(np.isfinite(lon)):
            raise IonosphereError("latitudes must lie in [-90, 90] and longitudes be finite")
        self.family = family
        self.alt = (float(alt_lo), float(alt_hi), int(nalt))
        self.lat, self.lon = lat, lon
        model = igrf_model or igrf.load()
        self.igrf_sha256 = model.sha256
        self.inclination, self.modip, self.dip_latitude = igrf.magnetic_coordinates(lat, lon, when, model)
        self._map_loader = map_loader
        self._maps = {}
        self._device = None

    def maps(self, month: int):
        if month not in self._maps:
            self._maps[month] = read_maps(month, self.family, loader=self._map_loader)
        return self._maps[month]

    def drivers(self, when: datetime, f107: float) -> dict:
        """Everything that is shared by the columns at ``when``."""
        when = when.replace(tzinfo=None)
        f107 = float(f107)
        if not np.isfinite(f107) or f107 <= 0.0:
            raise IonosphereError(f"F10.7 {f107} sfu is not a solar flux")
        r12 = f107_to_r12(f107)
        ig12 = r12_to_ig12(r12)
        ut = when.hour + when.minute / 60.0 + when.second / 3600.0
        d_f2, d_m3 = diurnal(NJ_F2, ut), diurnal(NJ_M3, ut)
        coef = np.zeros(NCOEF)
        for month, weight in month_weights(when):
            if weight == 0.0:
                continue
            u_f2, u_m3 = self.maps(month)
            f2 = d_f2 @ (u_f2[:, :, 0] * (1.0 - ig12 / 100.0) + u_f2[:, :, 1] * (ig12 / 100.0))
            m3 = d_m3 @ (u_m3[:, :, 0] * (1.0 - r12 / 100.0) + u_m3[:, :, 1] * (r12 / 100.0))
            coef[:NK_F2] += weight * f2
            coef[NK_F2:] += weight * m3
        sun_lat, sun_lon = subsolar_point(when)
        return {"coef": coef, "r12": r12, "ig12": ig12, "f107": f107, "sun_lat": sun_lat,
                "sun_lon": sun_lon, "season": season(when.month)}

    def evaluate(self, when: datetime, f107: float) -> dict:
        """Device arrays (ncol,) of :data:`OUTPUTS` at ``when``."""
        import cupy as cp
        from woof.core.kernels import get_kernel

        drv = self.drivers(when, f107)
        if self._device is None:
            self._device = tuple(cp.asarray(a, dtype=cp.float32)
                                 for a in (self.lat, self.lon, self.modip, self.dip_latitude))
        ncol = self.lat.size
        out = {k: cp.zeros(ncol, dtype=cp.float32) for k in OUTPUTS}
        coef = cp.asarray(drv["coef"], dtype=cp.float32)
        alt_lo, alt_hi, nalt = self.alt
        threads = 128
        get_kernel("spacewx_ionosphere", "spacewx_ionosphere")(
            ((ncol + threads - 1) // threads,), (threads,),
            (*self._device, coef, np.float32(drv["r12"]), np.float32(drv["f107"]),
             np.float32(drv["sun_lat"]), np.float32(drv["sun_lon"]), np.int32(drv["season"]),
             np.float32(alt_lo), np.float32(alt_hi), np.int32(nalt),
             *(out[k] for k in OUTPUTS), np.int64(ncol)))
        return out

    def reference(self, when: datetime, f107: float) -> dict:
        """The same model in NumPy float64 (the device kernel's test oracle)."""
        drv = self.drivers(when, f107)
        return profile_parameters(drv, self.lat, self.lon, self.modip, self.dip_latitude, self.alt)


def _epstein(nm, hm, b, h):
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        alpha = np.where(b != 0.0, (h - hm) / np.where(b != 0.0, b, 1.0), 30.0)
        e = np.exp(np.clip(alpha, -80.0, 80.0))
        return np.where(alpha > 25.0, 0.0, 4.0 * nm * e / (1.0 + e) ** 2)


def _epstein_top(nm, hm, b, h):
    g, r = 0.125, 100.0
    dh = h - hm
    z = dh / (b * (1.0 + r * g * dh / (r * b + g * dh)))
    e = np.exp(np.clip(z, -80.0, 80.0))
    return np.where(e > 1.0e11, 4.0 * nm / e, 4.0 * nm * e / (e + 1.0) ** 2)


def profile_parameters(drv, lat_deg, lon_deg, modip_deg, diplat_deg, alt) -> dict:
    """NumPy float64 version of the device kernel (same equations, same order)."""
    coef = drv["coef"]
    r12, f107 = drv["r12"], drv["f107"]
    fo = coef[:NK_F2] @ geographic(Q_F2, lat_deg, lon_deg, modip_deg)
    m3 = coef[NK_F2:] @ geographic(Q_M3, lat_deg, lon_deg, modip_deg)
    fo = np.maximum(fo, 0.0)
    lat = np.deg2rad(lat_deg)
    slat = np.deg2rad(drv["sun_lat"])
    cchi = np.clip(np.sin(slat) * np.sin(lat)
                   + np.cos(slat) * np.cos(lat) * np.cos(np.deg2rad(drv["sun_lon"]) - np.deg2rad(lon_deg)), -1.0, 1.0)
    chi = np.rad2deg(np.arccos(cchi))
    ee = np.exp(np.clip(12.0 * (chi - 86.23292796211615), -80.0, 80.0))
    f1c = 90.0 - 0.24 * np.exp(np.clip(20.0 - 0.2 * chi, -80.0, 80.0))
    chi_eff = (f1c * ee + chi) / (ee + 1.0)
    es = np.exp(np.deg2rad(0.3 * lat_deg))
    seasp = drv["season"] * (es - 1.0) / (es + 1.0)
    ce = np.maximum(0.0, np.cos(np.deg2rad(chi_eff)))
    foe = np.sqrt(0.49 + (1.112 - 0.019 * seasp) ** 2 * np.sqrt(f107) * ce**0.6)
    g1 = 0.00232 * r12 + 0.222
    g2 = 1.0 - r12 / 150.0 * np.exp(-(modip_deg / 40.0) ** 2)
    g3 = 1.2 - 0.0116 * np.exp(r12 / 41.84)
    g4 = 0.096 * (r12 - 25.0) / 150.0
    ratio = np.maximum(fo / foe, 1.7)
    hmf2 = 1490.0 / (m3 + g1 * g2 / (ratio - g3) + g4) - 176.0
    hme = 110.0
    nmf2 = np.maximum(1.24e10 * fo**2, 1.0e6)
    nme = np.maximum(1.24e10 * foe**2, 1.0e6)
    dndh = 0.01 * np.exp(-3.467 + 1.714 * np.log(np.maximum(fo, 1.0e-3)) + 2.02 * np.log(np.maximum(m3, 1.0e-3)))
    bbot = 3.85e-12 * nmf2 / dndh
    k = 3.22 - 0.0538 * fo - 0.00664 * hmf2 + 0.113 * hmf2 / bbot + 0.00257 * r12
    x = (k * bbot - 150.0) / 100.0
    btop = (100.0 * x + 150.0) / (0.041163 * x**2 - 0.183981 * x + 1.424472)
    btop = np.where(btop > 0.0, btop, 30.0)
    mdl = np.abs(diplat_deg)
    n_exp = 0.093 + 0.0046 * mdl - 0.000054 * mdl**2 + 0.0003 * r12
    f100 = 5.348 + 0.011 * mdl - 0.00023 * mdl**2
    f0 = 4.35 + 0.0058 * mdl - 0.00012 * mdl**2
    fs = f0 + (f100 - f0) * r12 / 100.0
    raw_f1 = np.where(cchi > 0.0, fs * np.maximum(cchi, 0.0) ** n_exp, 0.0)
    mult = np.minimum(-10.0 + 30.0 * cchi, 10.0) / 10.0
    has_f1 = mult >= 0.0
    fof1 = np.where(has_f1, mult * raw_f1 + (1.0 - mult) * foe, 0.0)
    nmf1 = np.where(has_f1, np.maximum(1.24e10 * fof1**2, 1.0), 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        b = 2.0 - 4.0 * nmf2 / nmf1
        d = b * b - 4.0
        root = np.where(d >= 0.0, 0.5 * (-b - np.sqrt(np.maximum(d, 0.0))), 0.0)
        hmf1 = np.where(root > 0.0, bbot * np.log(np.where(root > 0.0, root, 1.0)) + hmf2, 0.0)
    has_f1 &= hmf1 > 110.0
    bf1 = np.where(has_f1, 0.5 * (hmf1 - hme), 10.0)
    bf1 = np.where(bf1 > 0.0, bf1, 10.0)
    alt_lo, alt_hi, nalt = alt
    dz = (alt_hi - alt_lo) / nalt
    tec = np.zeros_like(fo)
    for i in range(nalt):
        h = alt_lo + (i + 0.5) * dz
        top = _epstein_top(nmf2, hmf2, btop, h)
        bottom_e = _epstein(nme, hme, 5.0, h)
        upper = np.where(has_f1, hmf1, hmf2)
        span = upper - hme
        u1 = (h - hme) / span
        u2 = (upper - h) / span
        mixed = (_epstein(nme, hme, 7.0, h) * (1.0 - u1**4)
                 + _epstein(np.where(has_f1, nmf1, nmf2), upper, np.where(has_f1, bf1, bbot), h) * (1.0 - u2**4))
        f2_bottom = _epstein(nmf2, hmf2, bbot, h)
        ne = np.where(h >= hmf2, top,
                      np.where(h <= hme, bottom_e,
                               np.where(has_f1 & (h >= hmf1), f2_bottom, mixed)))
        tec += np.maximum(ne, 1.0)
    return {"fof2": fo, "m3000": m3, "hmf2": hmf2, "nmf2": nmf2, "foe": foe,
            "fof1": np.where(has_f1, fof1, 0.0), "hmf1": np.where(has_f1, hmf1, 0.0),
            "b_bot": bbot, "b_top": btop, "tec": tec * dz * 1.0e-13}


def receipt(iono: Ionosphere, f107_source: str, f107_mode: str) -> dict:
    """What a run records about its ionosphere inputs."""
    return {"spacewx_maps": iono.family, "spacewx_maps_commit": PYIRI_COMMIT,
            "spacewx_igrf_sha256": iono.igrf_sha256, "spacewx_f107_source": f107_source,
            "spacewx_f107_mode": f107_mode,
            "spacewx_tec_range_km": [iono.alt[0], iono.alt[1]], "spacewx_tec_cells": iono.alt[2],
            "spacewx_columns_sha256": hashlib.sha256(
                np.stack([iono.lat, iono.lon]).astype(np.float64).tobytes()).hexdigest()}


def grid(dlat: float, dlon: float):
    """Global lat/lon nodes in the IONEX layout: 87.5 N to 87.5 S, 180 W to 180 E."""
    lat = np.arange(87.5, -87.5 - 1e-9, -abs(dlat))
    lon = np.arange(-180.0, 180.0 + 1e-9, abs(dlon))
    return lat, lon


__all__ = ["Ionosphere", "IonosphereError", "OUTPUTS", "FAMILIES", "read_maps", "subsolar_point",
           "month_weights", "f107_to_r12", "r12_to_ig12", "profile_parameters", "receipt", "grid"]
