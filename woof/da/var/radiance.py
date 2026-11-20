"""Clear-sky satellite radiances in 3D-Var (plan V4.3; docs/dev/WRFDA-3DVAR-PLAN.md section 13).

Opt-in.  Nothing here runs unless a caller builds a :class:`RadianceTerm`;
CRTM (``woof.rtm``, CRTM v2.4+, CC0) is imported only inside
:func:`radiance_term`, never at module level.

The pieces, in WRFDA's order (``da_get_innov_vector_radiance``,
``da_get_innov_vector_crtm``, ``da_qc_rad``):

1. **Records.**  :class:`RadianceObs` is one CRTM sensor's observations.
   :func:`from_imager_scene` splits a :class:`woof.obs.imager_scene.ImagerScene`
   (SatWeave, GOES ABI, AHI, SEVIRI readers) by sensor; :func:`from_footprints`
   takes a microwave sounder batch with the attributes of WIS2's
   ``MwFootprints`` (duck-typed: nothing here imports the WIS2 reader).
2. **Domain and thinning.**  The observation's grid position is WRFDA's
   (``da_llxy`` on the first guess's Lambert projection); thinning is GSI's
   ``makegrids``/``map2grids`` as WRFDA calls them (:func:`gsi_thin`).
3. **Background columns** (:func:`background_columns`): WRFDA's CRTM
   atmosphere at the observation (bilinear in ``da_interp_2d_partial``'s
   order, layers top first, interfaces interpolated in sigma, surface type
   by ``da_detsurtyp``).
4. **H(x_b) and the K-matrix** from CRTM, once per outer loop, binary64.
5. **QC** (:func:`qc_wrfda`): ``radiance_info``'s channel use and errors,
   then ``da_qc_goesabi``/``da_qc_ahi``/``da_qc_seviri``'s clear-sky tests
   (the Zhuge-Zou cloud test, off by default in WRFDA, is not ported).
6. **The term** (:class:`RadianceTerm`): the :class:`~woof.da.var.cost.ObsTerm`
   whose TL is WRFDA's K-matrix branch of ``da_transform_xtoy_crtm`` and
   whose AD is its exact transpose, scattered in a fixed serial order.

Derived from WRFDA 4.7.1 (``var/da/da_radiance``: ``da_get_innov_vector_crtm``,
``da_detsurtyp``, ``da_transform_xtoy_crtm(_adj)``, ``da_qc_goesabi``,
``da_qc_ahi``, ``da_qc_seviri``, ``gsi_thinning``; ``var/run/radiance_info``),
public domain; see ``licenses/LICENSE-WRF-public-domain.txt``.  WRF's
bundled CRTM 2.3.0 is never used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping, Sequence

import numpy as np

__all__ = [
    "QC_BAD",
    "QC_GOOD",
    "SATINFO",
    "Columns",
    "RadianceBatch",
    "RadianceObs",
    "RadianceQC",
    "RadianceTerm",
    "background_columns",
    "family",
    "from_footprints",
    "from_imager_scene",
    "gsi_thin",
    "locate",
    "qc_wrfda",
    "radiance_term",
    "read_surface",
]

QC_GOOD = 1
QC_BAD = -1
GRAVITY = 9.81
#: WRFDA zeroes the humidity TL above this pressure (hPa).
Q_TL_TOP_HPA = 75.0
#: CRTM's transmittance fits stop at this zenith (woof.rtm.crtm.MAX_SENSOR_ZENITH_DEG).
MAX_ZENITH_DEG = 65.0
#: The error WRFDA gives a rejected channel (da_qc_seviri); it never enters Jo.
REJECTED_ERROR_K = 500.0


# --------------------------------------------------------------------------- radiance_info

def _info(errors: Mapping[int, float], use: Sequence[int], window: int | None) -> dict:
    return {"error": dict(errors), "use": frozenset(use), "window": window}


#: WRFDA 4.7.1 ``var/run/radiance_info`` per sensor family, in CRTM channel
#: numbers: the ``varch`` error (K) and the channels with ``iuse = 1``.
#: ABI from ``goes-16-abi.info``, AHI from ``himawari-8-ahi.info``, SEVIRI
#: from ``msg-4-seviri.info`` (WRFDA numbers SEVIRI's IR channels 1-8; CRTM
#: numbers them 4-11).  ``window`` is the 11.2 um channel the optional
#: window screen uses.
SATINFO = {
    "abi": _info({7: 2.72, 8: 1.79, 9: 1.92, 10: 1.74, 11: 5.0, 12: 2.79, 13: 3.08, 14: 3.06, 15: 2.82,
                  16: 1.74}, (8, 9, 10), 14),
    "ahi": _info({7: 1.052, 8: 1.70, 9: 1.70, 10: 1.35, 11: 0.814, 12: 0.931, 13: 0.871, 14: 0.926,
                  15: 0.933, 16: 0.787}, (8,), 14),
    "seviri": _info({4: 1.8, 5: 2.5, 6: 2.25, 7: 1.25, 8: 1.25, 9: 1.25, 10: 1.45, 11: 1.25},
                    (5, 6, 7, 10), 9),
}
#: Families whose WRFDA QC rejects mixed surfaces and cloudy backgrounds
#: (da_qc_goesabi, da_qc_ahi); SEVIRI's has only the gross checks.
_SURFACE_QC = ("abi", "ahi")


def family(sensor: str) -> str:
    """``abi_g16`` -> ``abi``; ``amsua_metop-b`` -> ``amsua``."""
    return str(sensor).split("_", 1)[0]


# --------------------------------------------------------------------------- records

@dataclass
class RadianceObs:
    """One CRTM sensor's brightness temperatures at ``n`` locations.

    ``bt`` is ``(n, nchan)`` K, NaN where missing; ``channels`` are the
    instrument channel numbers as CRTM numbers them.  Angles are degrees.
    ``scanpos`` is the field of view of a scanning sounder (a VarBC
    predictor), 0 for imagers.  ``qc`` and ``error`` are optional ``(n,
    nchan)`` arrays from the reader's own screens (WIS2's MW screens): a
    location/channel with ``qc < 0`` stays rejected, and a finite ``error``
    replaces ``radiance_info``'s.
    """

    sensor: str
    channels: np.ndarray
    bt: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    sensor_zenith: np.ndarray
    sensor_azimuth: np.ndarray
    sensor_scan: np.ndarray
    solar_zenith: np.ndarray
    solar_azimuth: np.ndarray
    time: datetime
    scanpos: np.ndarray | None = None
    qc: np.ndarray | None = None
    error: np.ndarray | None = None
    provenance: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.channels = np.asarray(self.channels, dtype=np.int64).ravel()
        self.bt = np.asarray(self.bt, dtype=np.float64)
        n = self.bt.shape[0]
        if self.bt.shape != (n, self.channels.size):
            raise ValueError(f"bt has shape {self.bt.shape}, want (n, {self.channels.size})")
        for name in ("lat", "lon", "sensor_zenith", "sensor_azimuth", "sensor_scan", "solar_zenith",
                     "solar_azimuth"):
            a = np.asarray(getattr(self, name), dtype=np.float64).ravel()
            if a.size == 1 and n != 1:
                a = np.full(n, float(a[0]))
            if a.shape != (n,):
                raise ValueError(f"{name} has {a.size} values for {n} locations")
            setattr(self, name, a)
        self.scanpos = (np.zeros(n) if self.scanpos is None
                        else np.asarray(self.scanpos, dtype=np.float64).reshape(n))
        for name in ("qc", "error"):
            a = getattr(self, name)
            if a is not None:
                a = np.asarray(a, dtype=np.int64 if name == "qc" else np.float64)
                if a.shape != self.bt.shape:
                    raise ValueError(f"{name} has shape {a.shape}, want {self.bt.shape}")
                setattr(self, name, a)

    @property
    def n(self) -> int:
        return int(self.bt.shape[0])

    def subset(self, index) -> "RadianceObs":
        index = np.asarray(index)
        pick = {k: getattr(self, k)[index] for k in ("bt", "lat", "lon", "sensor_zenith", "sensor_azimuth",
                                                      "sensor_scan", "solar_zenith", "solar_azimuth", "scanpos")}
        for k in ("qc", "error"):
            pick[k] = None if getattr(self, k) is None else getattr(self, k)[index]
        return RadianceObs(sensor=self.sensor, channels=self.channels, time=self.time,
                           provenance=dict(self.provenance), **pick)


def from_imager_scene(scene, bands: Sequence[int] = (6200, 6900, 7330, 11200), *, bbox=None) -> list[RadianceObs]:
    """One :class:`RadianceObs` per CRTM sensor in ``scene``.

    ``bands`` are SatWeave wavelengths (nm); a sensor family that lacks one
    (SEVIRI has no 6.9 um) simply carries fewer channels.  Pixels with no
    attributed satellite, a limb or out-of-range flag, or no finite value in
    any chosen band are left out.  ``bbox = (south, north, west, east)``
    (degrees) cuts the scene first.
    """
    from woof.obs.imager_scene import FLAG_LIMB, FLAG_NO_SATELLITE, FLAG_OUT_OF_RANGE, band

    lat, lon = np.asarray(scene.lat, float), np.asarray(scene.lon, float)
    if lat.ndim == 1:
        lon, lat = np.meshgrid(lon, lat)
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    keep = (np.asarray(scene.platform_index) >= 0) & np.isfinite(lat) & np.isfinite(lon)
    keep &= (np.asarray(scene.flags) & (FLAG_LIMB | FLAG_NO_SATELLITE | FLAG_OUT_OF_RANGE)) == 0
    if bbox is not None:
        s, nn, w, e = (float(v) for v in bbox)
        keep &= (lat >= s) & (lat <= nn) & (lon >= w) & (lon <= e)
    out = []
    for p, plat in enumerate(scene.platforms):
        rows = np.flatnonzero((keep & (np.asarray(scene.platform_index) == p)).ravel())
        chosen = [nm for nm in bands if nm in scene.values and plat.family in band(nm).channels]
        if rows.size == 0 or not chosen:
            continue
        bt = np.stack([np.asarray(scene.values[nm], float).ravel()[rows] for nm in chosen], axis=1)
        finite = np.isfinite(bt).any(axis=1)
        rows, bt = rows[finite], bt[finite]
        if rows.size == 0:
            continue

        def flat(a):
            return np.asarray(a, float).ravel()[rows]

        out.append(RadianceObs(
            sensor=plat.crtm_sensor, channels=[band(nm).channels[plat.family] for nm in chosen], bt=bt,
            lat=lat.ravel()[rows], lon=lon.ravel()[rows], sensor_zenith=flat(scene.view_zenith),
            sensor_azimuth=flat(scene.view_azimuth), sensor_scan=flat(scene.scan_angle),
            solar_zenith=flat(scene.solar_zenith), solar_azimuth=flat(scene.solar_azimuth), time=scene.time,
            provenance={"reader": scene.provenance.get("source", type(scene).__name__), "platform": plat.name,
                        "bands_nm": list(chosen), "pixels": int(rows.size)}))
    return out


def from_footprints(fp, *, time: datetime | None = None, qc=None, error=None, settings=None) -> list[RadianceObs]:
    """Microwave sounder footprints (WIS2 ``MwFootprints``-like) as one record per CRTM sensor.

    ``fp`` needs ``sensor`` (per footprint), ``channels``, ``bt``, ``lat``,
    ``lon``, ``fov``, ``scan_angle``, ``sensor_zenith``, ``sensor_azimuth``,
    ``solar_zenith``, ``solar_azimuth`` and, unless ``time`` is given,
    ``time`` (datetime64 per footprint; the first one is used).  ``qc`` and
    ``error`` are the reader's own screens, ``(n, nchan)``.

    ``settings`` (a :class:`woof.da.obs_mw.MwSettings`, WIS2's MW adapter)
    supplies both when they are not given: WRFDA's assimilated channels and
    errors (``radiance_info``), and the limb scan positions refused
    (``LIMB_POSITIONS``).  Its static ``bias_k`` is not used; VarBC replaces it.
    The cloud and scattering screens that need the background are not
    applied here (``qc_wrfda``'s gross checks are).
    """
    if settings is not None and (qc is None or error is None):
        from woof.da.obs_mw import LIMB_POSITIONS

        inst = str(fp.instrument)
        chans = np.asarray(fp.channels, dtype=np.int64).ravel()
        chosen = set(settings.chosen(inst))
        fov = np.asarray(fp.fov)
        lo, hi = LIMB_POSITIONS[inst]
        limb = (fov <= lo) | (fov >= hi)
        if qc is None:
            qc = np.where(limb[:, None] | ~np.isin(chans, sorted(chosen))[None, :], -1, 0)
        if error is None:
            row = np.array([settings.error(inst, int(c)) if int(c) in chosen else np.nan for c in chans])
            error = np.broadcast_to(row, (fov.size, chans.size)).copy()
    sensors = np.asarray(fp.sensor).astype(str)
    if time is None:
        t = np.asarray(fp.time).astype("datetime64[s]")
        from datetime import timezone

        time = t.min().astype(datetime).replace(tzinfo=timezone.utc) if t.size else None
    out = []
    for s in sorted(set(sensors.tolist())):
        rows = np.flatnonzero(sensors == s)
        sub = (lambda a: None if a is None else np.asarray(a)[rows])
        out.append(RadianceObs(
            sensor=s, channels=fp.channels, bt=np.asarray(fp.bt, float)[rows], lat=np.asarray(fp.lat)[rows],
            lon=np.asarray(fp.lon)[rows], sensor_zenith=np.asarray(fp.sensor_zenith)[rows],
            sensor_azimuth=np.asarray(fp.sensor_azimuth)[rows], sensor_scan=np.asarray(fp.scan_angle)[rows],
            solar_zenith=np.asarray(fp.solar_zenith)[rows], solar_azimuth=np.asarray(fp.solar_azimuth)[rows],
            scanpos=np.asarray(fp.fov, float)[rows], time=time, qc=sub(qc), error=sub(error),
            provenance={"reader": type(fp).__name__, "footprints": int(rows.size)}))
    return out


# --------------------------------------------------------------------------- location and thinning

def locate(proj, lat, lon) -> dict:
    """WRFDA's grid position of each location, and which are inside the domain.

    ``proj`` is :class:`~woof.da.var.obs_wrfda.LambertWRFDA`.  Returns 0-based
    ``i0``, ``j0`` and the weights ``dx``, ``dy``, ``dxm``, ``dym``
    (``da_togrid``) for the locations with ``1 <= x < ide`` and ``1 <= y <
    jde`` (all four corners in the domain), and ``inside`` (bool).
    """
    lat = np.asarray(lat, float).ravel()
    lon = np.asarray(lon, float).ravel()
    n = lat.size
    xs = np.full(n, np.nan)
    ys = np.full(n, np.nan)
    # A vectorised first pass (platform libm) with a 2-cell margin picks the
    # candidates; their position is then WRFDA's, from the correctly rounded llxy.
    rough = _llxy_numpy(proj, lat, lon)
    cand = np.flatnonzero((rough[0] > -1.0) & (rough[0] < proj.ide + 2.0)
                          & (rough[1] > -1.0) & (rough[1] < proj.jde + 2.0))
    for k in cand.tolist():
        xs[k], ys[k] = proj.llxy(float(lat[k]), float(lon[k]))
    inside = np.isfinite(xs) & (xs >= 1.0) & (xs < proj.ide) & (ys >= 1.0) & (ys < proj.jde)
    i = np.where(inside, np.floor(np.where(inside, xs, 1.0)), 1.0).astype(np.int64)
    j = np.where(inside, np.floor(np.where(inside, ys, 1.0)), 1.0).astype(np.int64)
    dx = np.where(inside, xs - i, 0.0)
    dy = np.where(inside, ys - j, 0.0)
    return {"inside": inside, "i0": i - 1, "j0": j - 1, "dx": dx, "dy": dy, "dxm": 1.0 - dx, "dym": 1.0 - dy,
            "x": xs, "y": ys}


def _llxy_numpy(proj, lat, lon):
    rad = math.pi / 180.0
    deltalon = lon - proj.stand_lon
    deltalon = np.where(deltalon > 180.0, deltalon - 360.0, deltalon)
    deltalon = np.where(deltalon < -180.0, deltalon + 360.0, deltalon)
    ctl1r = math.cos(proj.truelat1 * rad)
    with np.errstate(all="ignore"):
        ratio = np.tan((90.0 * proj.hemi - lat) * rad / 2.0) / math.tan((90.0 * proj.hemi - proj.truelat1) * rad / 2.0)
        rm = proj.rebydx * ctl1r / proj.cone * ratio ** proj.cone
    arg = proj.cone * (deltalon * rad)
    x = proj.polei + proj.hemi * rm * np.sin(arg)
    y = proj.polej - rm * np.cos(arg)
    if proj.hemi == -1.0:
        x, y = 2.0 - x, 2.0 - y
    return x + proj.knowni - 1.0, y + proj.knownj - 1.0


#: GSI's constants (gsi_constants.f90) as WRFDA's thinning uses them.
_REARTH_EQUATOR = 6.37813662e6
_PI = math.acos(-1.0)
_DEG2RAD = _PI / 180.0


def _grdcrd(d: np.ndarray, x: np.ndarray) -> np.ndarray:
    """GSI's ``grdcrd(d, nd, x, nx, flg=1)``: the 1-based fractional position of ``d`` in ``x``."""
    nx = x.size
    if nx == 1:
        return np.ones_like(d)
    # isrchf(nx-1, x, d, 1): the first k with d <= x(k), else nx; ix = that - 1, and 1 if d <= x(1)
    ix = np.maximum(np.searchsorted(x[:nx - 1], d, side="left"), 1)
    lo, hi = x[ix - 1], x[ix]
    return ix.astype(np.float64) + (d - lo) / (hi - lo)


def gsi_thin(lat, lon, mesh_km: float, domain_lat, domain_lon) -> np.ndarray:
    """GSI's ``makegrids`` and ``map2grids`` with ``crit = 1``: the kept indices, ascending.

    The thinning boxes are an equal-area latitude-longitude grid of
    ``mesh_km`` over the domain's latitude and longitude range (``rlat_min``,
    ``dlat_grid``... of ``da_setup_obs_structures``).  In each box the
    observation with the smallest ``dist1 = dxx^2 + dyy^2 + 1/2`` (its offset
    from the box centre in box units) is kept; the first one wins a tie, as
    ``map2grids``' strict ``crit < score_crit`` has it.
    """
    lat = np.asarray(lat, float).ravel()
    lon = np.asarray(lon, float).ravel()
    dlat_all = np.asarray(domain_lat, float).ravel()
    dlon_all = np.asarray(domain_lon, float).ravel()
    dlon360 = np.where(dlon_all < 0.0, 360.0 + dlon_all, dlon_all)
    rlat_min, rlat_max = float(dlat_all.min()), float(dlat_all.max())
    rlon_min, rlon_max = float(dlon360.min()), float(dlon360.max())
    dlat_grid, dlon_grid = rlat_max - rlat_min, rlon_max - rlon_min
    # makegrids
    rkm2dg = 360.0 / (2.0 * _PI * _REARTH_EQUATOR) * 1000.0
    dx = mesh_km * rkm2dg
    dy = dx
    mlat = int(dlat_grid / dy + 0.5)
    mlonx = int(dlon_grid / dx + 0.5)
    delat = dlat_grid / max(mlat, 1)  # WRFDA divides by mlat before clamping it to 2
    dgv = delat * 0.5
    mlat = max(2, mlat)
    mlonx = max(2, mlonx)
    glat = np.empty(mlat)
    mlon = np.empty(mlat, dtype=np.int64)
    glon = []
    hll = []
    count = 0
    for jj in range(mlat):
        g = (rlat_min + jj * delat) * _DEG2RAD
        glatm = g + dgv * _DEG2RAD
        factor = abs(math.cos(abs(glatm)))
        mlonj = max(2, _nint(mlonx * factor))
        mlon[jj] = mlonj
        delon = dlon_grid / mlonj
        glat[jj] = min(max(-_PI * 0.5, g), _PI * 0.5)
        row = np.array([min(max(0.0, (rlon_min + ii * delon) * _DEG2RAD), 2.0 * _PI) for ii in range(mlonj)])
        glon.append(row)
        hll.append(np.arange(count, count + mlonj))
        count += mlonj
    # map2grids, observation by observation in input order
    dlat1 = _grdcrd(lat * _DEG2RAD, glat)
    iy = np.trunc(dlat1).astype(np.int64)
    fy = dlat1 - iy
    iy = np.clip(iy, 1, mlat)
    lon360 = np.where(lon < 0.0, lon + 360.0, lon) * _DEG2RAD
    fx = np.empty(lat.size)
    box = np.empty(lat.size, dtype=np.int64)
    for row in np.unique(iy):
        sel = np.flatnonzero(iy == row)
        dlon1 = _grdcrd(lon360[sel], glon[row - 1])
        ix = np.trunc(dlon1).astype(np.int64)
        fx[sel] = dlon1 - ix
        ix = np.clip(ix, 1, mlon[row - 1])
        box[sel] = hll[row - 1][ix - 1]
    dxx = 0.5 - np.minimum(fx, 1.0 - fx)
    dyy = 0.5 - np.minimum(fy, 1.0 - fy)
    dist1 = dxx * dxx + dyy * dyy + 0.5
    best: dict = {}
    for k, (b, c) in enumerate(zip(box.tolist(), dist1.tolist())):
        if b not in best or c < best[b][0]:
            best[b] = (c, k)
    return np.array(sorted(k for _, k in best.values()), dtype=np.int64)


def _nint(x: float) -> int:
    """Fortran ``nint``: round half away from zero."""
    return int(math.floor(x + 0.5)) if x >= 0 else -int(math.floor(-x + 0.5))


# --------------------------------------------------------------------------- background columns

def _hcol(f, h):
    """``da_interp_2d_partial``: ``dym*(dxm*f(i,j)+dx*f(i+1,j)) + dy*(dxm*f(i,j+1)+dx*f(i+1,j+1))``."""
    i0, j0, dx, dy, dxm, dym = h
    if f.ndim == 3:
        dx, dy, dxm, dym = (a[:, None] for a in (dx, dy, dxm, dym))
    return (dym * (dxm * f[i0, j0] + dx * f[i0 + 1, j0])
            + dy * (dxm * f[i0, j0 + 1] + dx * f[i0 + 1, j0 + 1]))


def _detsurtyp(h, landmask, xice, snow):
    """``da_detsurtyp``: the surface type ``isflg`` (0 sea, 1 ice, 2 land, 3 snow, 4-7 mixed) and fractions."""
    i0, j0, dx, dy, dxm, dym = h
    w = (dym * dxm, dym * dx, dy * dxm, dy * dx)

    def avg(f):
        s = w[0] * f[i0, j0] + w[1] * f[i0 + 1, j0] + w[2] * f[i0, j0 + 1] + w[3] * f[i0 + 1, j0 + 1]
        return 1.0 / (w[0] + w[1] + w[2] + w[3]) * s

    lndp = avg(np.asarray(landmask, float))
    icep = avg(np.asarray(xice, float))
    snop = avg(np.minimum(np.asarray(snow, float), 1.0))
    seap = np.where(icep > 0.0, 0.0, 1.0 - lndp)
    snop = np.where(icep > 0.0, np.minimum(snop, 1.0 - icep), snop)
    seap = np.where(icep > 0.0, 1.0 - icep - snop, seap)
    lndp = np.maximum(1.0 - seap - icep - snop, 0.0)
    isflg = np.full(lndp.shape, -1, dtype=np.int64)
    for flag, p in ((0, seap), (1, icep), (2, lndp), (3, snop)):
        isflg = np.where(p >= 0.99, flag, isflg)
    mixed = isflg < 0
    mix = np.where(seap > lndp,
                   np.where(seap > icep, np.where(seap > snop, 4, 7), np.where(icep > snop, 5, 7)),
                   np.where(lndp > icep, np.where(lndp > snop, 6, 7), np.where(icep > snop, 5, 7)))
    isflg = np.where(mixed, mix, isflg)
    return isflg, {"sea": seap, "ice": icep, "land": lndp, "snow": snop}


@dataclass
class Columns:
    """WRFDA's CRTM atmosphere at ``n`` observation locations (layers top first)."""

    h: tuple                     # (i0, j0, dx, dy, dxm, dym) per location
    pressure_hpa: np.ndarray     # (n, nz) layer pressure, top first
    level_pressure_hpa: np.ndarray  # (n, nz + 1) interfaces, ptop .. psfc
    temperature_k: np.ndarray    # (n, nz)
    q: np.ndarray                # (n, nz) specific humidity kg/kg (xb%q)
    mixing_ratio_gkg: np.ndarray  # (n, nz) 1000 q / (1 - q)
    skin_temperature_k: np.ndarray
    wind_speed_ms: np.ndarray
    wind_direction_deg: np.ndarray
    surface_altitude_m: np.ndarray
    isflg: np.ndarray
    fractions: dict
    clwp_kgm2: np.ndarray | None
    ok: np.ndarray               # (n,) False where the interfaces are not monotonic

    def profiles(self, index=None):
        """A :class:`woof.rtm.profiles.Profiles` for the locations ``index`` (default: all)."""
        from woof.rtm.profiles import Profiles, standard_ozone_ppmv

        sel = slice(None) if index is None else np.asarray(index)
        fr = {k: np.asarray(v)[sel] for k, v in self.fractions.items()}
        total = fr["sea"] + fr["ice"] + fr["land"] + fr["snow"]
        fr = {k: v / total for k, v in fr.items()}
        p = self.pressure_hpa[sel]
        return Profiles(level_pressure_hpa=self.level_pressure_hpa[sel], pressure_hpa=p,
                        temperature_k=self.temperature_k[sel], water_vapour_gkg=self.mixing_ratio_gkg[sel],
                        ozone_ppmv=standard_ozone_ppmv(p), land_fraction=fr["land"], water_fraction=fr["sea"],
                        snow_fraction=fr["snow"], ice_fraction=fr["ice"],
                        skin_temperature_k=self.skin_temperature_k[sel], wind_speed_ms=self.wind_speed_ms[sel],
                        wind_direction_deg=self.wind_direction_deg[sel],
                        surface_altitude_m=self.surface_altitude_m[sel])


def background_columns(xb: dict, surface: Mapping, h: tuple) -> Columns:
    """The background at the locations ``h`` as ``da_get_innov_vector_crtm`` builds it.

    ``xb`` is :func:`woof.da.var.analysis.wrftoxb`'s (``p``, ``t``, ``q``,
    ``psfc``, ``ptop``, ``sigmah``, ``sigmaf``, ``terr``, ``(nx, ny, nz)``).
    ``surface`` holds the first guess's ``TSK``, ``U10``, ``V10``,
    ``LANDMASK``, ``XICE`` and ``SNOWC`` ``(nx, ny)``, and optionally
    ``QCLOUD`` ``(nx, ny, nz)`` for the cloud-water-path test.
    """
    nz = xb["t"].shape[2]
    p = _hcol(np.asarray(xb["p"], float), h)[:, ::-1] * 0.01          # hPa, top first
    t = _hcol(np.asarray(xb["t"], float), h)[:, ::-1]
    q = _hcol(np.asarray(xb["q"], float), h)[:, ::-1]
    psfc = _hcol(np.asarray(xb["psfc"], float), h) * 0.01
    ptop = float(xb["ptop"]) * 0.01
    sigmah = np.asarray(xb["sigmah"], float)
    sigmaf = np.asarray(xb["sigmaf"], float)
    n = p.shape[0]
    lev = np.empty((n, nz + 1))
    lev[:, 0] = ptop
    lev[:, nz] = psfc
    for k in range(1, nz):  # WRFDA's k = kts..kte-1 (1-based), top first
        kk = nz - k  # 1-based model level kte-k
        lev[:, k] = ((p[:, k] - p[:, k - 1]) / (sigmah[kk - 1] - sigmah[kk])
                     * (sigmaf[kk] - sigmah[kk]) + p[:, k - 1])
    # CRTM's Profiles also needs each layer strictly inside its interfaces
    ok = (np.all(lev[:, 1:] > lev[:, :-1], axis=1) & np.all(p > lev[:, :-1], axis=1)
          & np.all(p < lev[:, 1:], axis=1))
    mr = 1000.0 * q / (1.0 - q)
    u10 = _hcol(np.asarray(surface["U10"], float), h)
    v10 = _hcol(np.asarray(surface["V10"], float), h)
    isflg, fractions = _detsurtyp(h, surface["LANDMASK"], surface["XICE"], surface["SNOWC"])
    clwp = None
    if surface.get("QCLOUD") is not None:
        qcw = np.maximum(_hcol(np.asarray(surface["QCLOUD"], float), h)[:, ::-1], 0.0)
        dpf = 100.0 * (lev[:, 1:] - lev[:, :-1])
        clw = np.where(p < 100.0, 0.0, qcw * dpf / GRAVITY)
        clwp = np.zeros(n)
        for k in range(nz):
            clwp = clwp + clw[:, k]
    return Columns(h=h, pressure_hpa=p, level_pressure_hpa=lev, temperature_k=t, q=q, mixing_ratio_gkg=mr,
                   skin_temperature_k=_hcol(np.asarray(surface["TSK"], float), h),
                   wind_speed_ms=np.hypot(u10, v10),
                   wind_direction_deg=np.mod(np.degrees(np.arctan2(-u10, -v10)), 360.0),
                   surface_altitude_m=_hcol(np.asarray(xb["terr"], float), h), isflg=isflg, fractions=fractions,
                   clwp_kgm2=clwp, ok=ok)


# --------------------------------------------------------------------------- QC

@dataclass(frozen=True)
class RadianceQC:
    """WRFDA's clear-sky QC switches (namelist ``&wrfvar14`` defaults) and woof's opt-in extras."""

    thinning: bool = True
    thinning_mesh_km: float = 60.0
    #: Channels to use per family, overriding ``radiance_info``'s ``iuse``.
    use_channels: Mapping | None = None
    #: Errors (K) per CRTM channel, overriding ``radiance_info``'s.
    error_k: Mapping | None = None
    error_factor: float = 1.0
    inv_gross_k: float = 15.0
    sigma_gross: float = 3.0
    clwp_max_kgm2: float = 0.2
    only_sea_rad: bool = False
    max_zenith_deg: float = MAX_ZENITH_DEG
    #: Opt-in (plan 11.5, not WRFDA): reject the location when the window
    #: channel's O-B is below ``-window_screen_k``.
    window_screen_k: float | None = None

    def used(self, sensor: str) -> frozenset:
        fam = family(sensor)
        if self.use_channels is not None and fam in self.use_channels:
            return frozenset(int(c) for c in self.use_channels[fam])
        return SATINFO.get(fam, {"use": frozenset()})["use"]

    def error(self, sensor: str, channel: int) -> float:
        if self.error_k is not None and int(channel) in self.error_k:
            return float(self.error_k[int(channel)]) * self.error_factor
        info = SATINFO.get(family(sensor))
        if info is None or int(channel) not in info["error"]:
            return math.nan
        return float(info["error"][int(channel)]) * self.error_factor


def qc_wrfda(sensor: str, channels, obs_bt, hx, inv, cols: Columns, settings: RadianceQC,
             reader_qc=None, reader_error=None) -> tuple[np.ndarray, np.ndarray, dict]:
    """``da_qc_goesabi``/``da_qc_ahi``/``da_qc_seviri``'s clear-sky decisions: ``(qc, error, counts)``.

    ``inv`` is O - B after bias correction (WRFDA's ``tb_inv`` after
    ``da_varbc_direct``).  Rejected entries get ``qc = -1`` and error 500 K.
    Families without a WRFDA geostationary QC (the MW sounders) get the
    reader's screens and the gross checks.
    """
    fam = family(sensor)
    channels = np.asarray(channels)
    n, nchan = inv.shape
    qc = np.full((n, nchan), QC_GOOD, dtype=np.int64)
    err = np.array([[settings.error(sensor, c) for c in channels]] * n, dtype=np.float64).reshape(n, nchan)
    if reader_error is not None:
        err = np.where(np.isfinite(reader_error), reader_error, err)
    counts = {}

    def reject(mask, name):
        mask = np.broadcast_to(mask, qc.shape) & (qc == QC_GOOD)
        counts[name] = int(mask.sum())
        qc[mask] = QC_BAD

    if reader_qc is not None:
        reject(np.asarray(reader_qc) < 0, "reader")
    used = settings.used(sensor)
    reject(~np.isin(channels, list(used))[None, :] if fam in SATINFO or settings.use_channels else
           np.zeros((1, nchan), bool), "iuse")
    reject(~np.isfinite(err), "no_error")
    reject(~np.isfinite(obs_bt) | (obs_bt < 0.0) | ~np.isfinite(hx) | (hx < 0.0), "missing")
    reject(~cols.ok[:, None], "pressure_not_monotonic")
    if fam in _SURFACE_QC:
        reject((cols.isflg >= 4)[:, None], "mixed_surface")
        if settings.only_sea_rad:
            wv = np.isin(channels, (8, 9, 10))[None, :]
            reject((cols.isflg != 0)[:, None] & ~wv, "only_sea")
        if cols.clwp_kgm2 is not None:
            reject((cols.clwp_kgm2 >= settings.clwp_max_kgm2)[:, None], "clwp")
    if settings.window_screen_k is not None:
        win = SATINFO.get(fam, {}).get("window")
        hit = np.flatnonzero(channels == win)
        if hit.size:
            reject((inv[:, hit[0]] < -float(settings.window_screen_k))[:, None], "window_screen")
    with np.errstate(invalid="ignore"):
        reject(np.abs(inv) > settings.inv_gross_k, "gross_15k")
        reject(np.abs(inv) > settings.sigma_gross * err, "gross_3sigma")
    err = np.where(qc == QC_GOOD, err, REJECTED_ERROR_K)
    counts["good"] = int((qc == QC_GOOD).sum())
    return qc, err, counts


# --------------------------------------------------------------------------- the term

@dataclass
class RadianceBatch:
    """One sensor after the innovation step: everything the TL/AD and VarBC need.

    ``kt`` and ``kr`` are CRTM's ``dBT/dT`` (K/K) and ``dBT/dr`` (K per g/kg),
    ``(n, nchan, nz)`` in CRTM's layer order (top first); ``q`` and
    ``pressure_hpa`` are the background at those layers.
    """

    sensor: str
    channels: np.ndarray
    h: tuple
    kt: np.ndarray
    kr: np.ndarray
    q: np.ndarray
    pressure_hpa: np.ndarray
    obs: np.ndarray        # (n, nchan) observed BT
    hx: np.ndarray         # (n, nchan) H(x_b)
    inv: np.ndarray        # (n, nchan) O - B, bias-corrected when VarBC applied it
    qc: np.ndarray         # (n, nchan)
    error: np.ndarray      # (n, nchan)
    extra: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.obs.shape[0])

    def tl_factors(self):
        """``dr = 1000 / (1 - q_b)^2 dq``, zero above 75 hPa (``da_transform_xtoy_crtm``)."""
        f = 1000.0 / (1.0 - self.q) ** 2
        return np.where(self.pressure_hpa < Q_TL_TOP_HPA, 0.0, f)


def _host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


class RadianceTerm:
    """The radiance :class:`~woof.da.var.cost.ObsTerm` over one or more :class:`RadianceBatch`.

    ``y`` is flattened batch by batch, location-major, channels within a
    location.  The TL is WRFDA's K-matrix branch::

        dBT(c) = sum_{l = 2..nz} Kt(c, l) dT(l) + Kr(c, l) (f(l) dq(l))

        f = 1000 / (1 - q_b)^2,  0 above 75 hPa

    accumulated over layers top first (WRFDA zeroes the top layer).  ``dT``
    and ``dq`` are ``xa%t`` and ``xa%q`` interpolated as
    ``da_interp_2d_partial`` does.  WRFDA's surface-pressure term
    (``ps_jacobian``) is absent: woof's CRTM shim returns no such Jacobian.

    The AD is the exact transpose: per location the column adjoint, then the
    four-corner scatter, location by location in order (no atomics).
    Arithmetic is on the host in the input's real type, so the words do not
    depend on the backend.
    """

    kind = "radiance"

    def __init__(self, batches: Sequence[RadianceBatch], shape: tuple):
        self.batches = list(batches)
        self.shape = tuple(int(s) for s in shape)  # (nx, ny, nz)
        sizes = [b.n * b.channels.size for b in self.batches]
        self.offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        cat = (lambda name: np.concatenate([getattr(b, name).ravel() for b in self.batches])
               if self.batches else np.zeros(0))
        self.d = np.asarray(cat("inv"), dtype=np.float64)
        self.d = np.where(np.isfinite(self.d), self.d, 0.0)
        self.err = np.asarray(cat("error"), dtype=np.float64)
        self.mask = np.asarray(cat("qc")) >= 0 if self.batches else np.zeros(0, bool)
        self._f = [b.tl_factors() for b in self.batches]
        #: The VarBC control block (:meth:`woof.da.var.varbc.VarBC.attach`), or None.
        self.varbc = None

    @property
    def control_size(self) -> int:
        """The size of this term's block of the control vector (VarBC's coefficients; 0 without)."""
        return 0 if self.varbc is None else self.varbc.size

    def jp(self, v) -> float:
        """``Jp`` of the block ``v`` (``cv + xhat``)."""
        return self.varbc.jp(v)

    def jp_grad(self, v):
        return self.varbc.jp_grad(v)

    @property
    def m(self) -> int:
        return int(self.offsets[-1])

    def jo_split(self):
        return [(self.kind, [(b.sensor, np.arange(self.offsets[k], self.offsets[k + 1]))
                             for k, b in enumerate(self.batches)])]

    def set_innovations(self, inv_per_batch: Sequence[np.ndarray]) -> None:
        """Replace each batch's ``inv`` (VarBC's bias update) and :attr:`d`."""
        for b, inv in zip(self.batches, inv_per_batch):
            b.inv = np.asarray(inv, dtype=np.float64).reshape(b.obs.shape)
        self.d = np.concatenate([np.where(np.isfinite(b.inv), b.inv, 0.0).ravel() for b in self.batches])

    # -------------------------------------------------------------- TL

    def tl(self, x: dict):
        t = _host(x["t"])
        dt = t.dtype
        q = _host(x["q"]).astype(dt, copy=False)
        out = []
        for b, f in zip(self.batches, self._f):
            h = tuple(a.astype(dt) if a.dtype.kind == "f" else a for a in b.h)
            ta = _hcol(t, h)[:, ::-1]                          # (n, nz) top first
            qa = (f.astype(dt) * _hcol(q, h)[:, ::-1])         # dr, g/kg
            kt, kr = b.kt.astype(dt), b.kr.astype(dt)
            y = np.zeros((b.n, b.channels.size), dtype=dt)
            for layer in range(1, ta.shape[1]):               # kts+1 .. kte
                y = y + kt[:, :, layer] * ta[:, None, layer] + kr[:, :, layer] * qa[:, None, layer]
            out.append(y)
        if self.varbc is not None and x.get("varbc") is not None:
            self.varbc.tl_add(out, _host(x["varbc"]))
        return np.concatenate([y.ravel() for y in out]) if out else np.zeros(0, dtype=dt)

    # -------------------------------------------------------------- AD

    def ad(self, y) -> dict:
        y = _host(y)
        dt = y.dtype
        nx, ny, nz = self.shape
        ft = np.zeros(nx * ny * nz, dtype=dt)
        fq = np.zeros(nx * ny * nz, dtype=dt)
        for k, (b, f) in enumerate(zip(self.batches, self._f)):
            yb = y[self.offsets[k]:self.offsets[k + 1]].reshape(b.n, b.channels.size)
            kt, kr = b.kt.astype(dt), b.kr.astype(dt)
            ta = np.zeros((b.n, nz), dtype=dt)
            ra = np.zeros((b.n, nz), dtype=dt)
            for c in range(b.channels.size):                  # channel by channel, in order
                ta[:, 1:] = ta[:, 1:] + kt[:, c, 1:] * yb[:, c, None]
                ra[:, 1:] = ra[:, 1:] + kr[:, c, 1:] * yb[:, c, None]
            qa = f.astype(dt) * ra
            # back to model levels (bottom first) and the four corners
            ta, qa = ta[:, ::-1], qa[:, ::-1]
            i0, j0, dx, dy, dxm, dym = b.h
            w = [a.astype(dt)[:, None] for a in (dym * dxm, dym * dx, dy * dxm, dy * dx)]
            lev = np.arange(nz)
            idx, vt, vq = [], [], []
            for (di, dj), ww in zip(((0, 0), (1, 0), (0, 1), (1, 1)), w):
                idx.append(((i0 + di) * ny + (j0 + dj))[:, None] * nz + lev[None, :])
                vt.append(ww * ta)
                vq.append(ww * qa)
            order = np.stack(idx, axis=1).ravel()              # (location, corner, level)
            np.add.at(ft, order, np.stack(vt, axis=1).ravel())
            np.add.at(fq, order, np.stack(vq, axis=1).ravel())
        out = {"t": ft.reshape(nx, ny, nz), "q": fq.reshape(nx, ny, nz)}
        if self.varbc is not None:
            out["varbc"] = self.varbc.ad([y[self.offsets[k]:self.offsets[k + 1]].reshape(b.n, b.channels.size)
                                          for k, b in enumerate(self.batches)])
        return out


# --------------------------------------------------------------------------- the door

#: The first-guess fields the CRTM surface needs beyond ``wrftoxb``'s.
SURFACE_FIELDS = ("TSK", "U10", "V10", "LANDMASK", "XICE", "SNOWC", "QCLOUD")


def read_surface(path) -> dict:
    """``SURFACE_FIELDS`` from a WRF first guess, first record, ``(i, j[, k])`` binary64.

    A missing ``U10``/``V10``/``XICE``/``SNOWC`` is zero; a missing
    ``QCLOUD`` skips the cloud-water test (WRFDA reads all of them).
    """
    import netCDF4

    out = {}
    with netCDF4.Dataset(path) as nc:
        nc.set_auto_mask(False)
        shape = None
        for name in SURFACE_FIELDS:
            if name in nc.variables:
                out[name] = np.asarray(nc.variables[name][0], dtype=np.float64).transpose()
                if out[name].ndim == 2:
                    shape = out[name].shape
    if "TSK" not in out or "LANDMASK" not in out:
        raise ValueError(f"{path}: the radiance door needs TSK and LANDMASK")
    for name in ("U10", "V10", "XICE", "SNOWC"):
        out.setdefault(name, np.zeros(shape))
    out.setdefault("QCLOUD", None)
    return out


def radiance_term(obs: Sequence[RadianceObs], xb: dict, surface: Mapping, proj, operator, *,
                  settings: RadianceQC | None = None, bias=None) -> tuple[RadianceTerm, dict]:
    """Innovations, QC and the term for each record in ``obs``, with an open CRTM ``operator``.

    ``operator`` is a :class:`woof.rtm.crtm.ClearSkyOperator` opened with
    every sensor in ``obs`` (or any object with its ``simulate``).  ``bias``
    is an optional callable ``bias(batch, columns) -> (n, nchan)`` bias in K
    subtracted from O - B before QC (VarBC's ``da_varbc_direct``; a static
    offset works too).  Returns the term and a provenance dict with the
    counts per sensor.
    """
    settings = settings or RadianceQC()
    shape = tuple(int(s) for s in np.shape(xb["t"]))
    batches, report = [], {"settings": {k: (dict(v) if isinstance(v, Mapping) else v)
                                         for k, v in settings.__dict__.items()}, "sensors": {},
                           "omitted": ["surface-pressure Jacobian (not in woof's CRTM shim)"]}
    for rec in obs:
        counts = {"read": rec.n}
        geometry_ok = np.ones(rec.n, bool)
        for a in (rec.lat, rec.lon, rec.sensor_scan, rec.sensor_azimuth, rec.solar_zenith, rec.solar_azimuth):
            geometry_ok &= np.isfinite(a)
        counts["geometry_missing"] = int((~geometry_ok).sum())
        keep = geometry_ok & np.isfinite(rec.sensor_zenith) & (rec.sensor_zenith <= settings.max_zenith_deg)
        counts["zenith"] = int((geometry_ok & ~keep).sum())
        rec = rec.subset(np.flatnonzero(keep))
        loc = locate(proj, rec.lat, rec.lon)
        counts["outside_domain"] = int((~loc["inside"]).sum())
        rec, loc = rec.subset(np.flatnonzero(loc["inside"])), {k: v[loc["inside"]] for k, v in loc.items()}
        if settings.thinning and rec.n:
            if xb.get("lat") is None or xb.get("lon") is None:
                raise ValueError("thinning needs the first guess's XLAT and XLONG (xb lat/lon)")
            kept = gsi_thin(rec.lat, rec.lon, settings.thinning_mesh_km, xb["lat"], xb["lon"])
            counts["thinned"] = int(rec.n - kept.size)
            rec, loc = rec.subset(kept), {k: v[kept] for k, v in loc.items()}
        if rec.n == 0:
            report["sensors"][rec.sensor] = counts
            continue
        h = (loc["i0"], loc["j0"], loc["dx"], loc["dy"], loc["dxm"], loc["dym"])
        cols = background_columns(xb, surface, h)
        good = np.flatnonzero(cols.ok)
        counts["pressure_not_monotonic"] = int(rec.n - good.size)
        nz = shape[2]
        nchan = rec.channels.size
        hx = np.full((rec.n, nchan), np.nan)
        kt = np.zeros((rec.n, nchan, nz))
        kr = np.zeros((rec.n, nchan, nz))
        if good.size:
            geometry = dict(sensor_zenith=rec.sensor_zenith[good], sensor_scan=rec.sensor_scan[good],
                            sensor_azimuth=rec.sensor_azimuth[good], solar_zenith=rec.solar_zenith[good],
                            solar_azimuth=rec.solar_azimuth[good], latitude=rec.lat[good],
                            longitude=rec.lon[good])
            sim = operator.simulate(rec.sensor, cols.profiles(good), geometry, rec.time,
                                    channels=rec.channels.tolist(), jacobian=True)
            hx[good] = sim.brightness_temperature
            kt[good] = sim.k_temperature
            kr[good] = sim.k_water_vapour
        batch = RadianceBatch(sensor=rec.sensor, channels=rec.channels, h=h, kt=kt, kr=kr, q=cols.q,
                              pressure_hpa=cols.pressure_hpa, obs=rec.bt, hx=hx, inv=rec.bt - hx,
                              qc=np.zeros((rec.n, nchan), np.int64), error=np.zeros((rec.n, nchan)),
                              extra={"columns": cols, "record": rec})
        if bias is not None:
            batch.inv = batch.inv - np.asarray(bias(batch, cols), dtype=np.float64)
        qc, err, qcounts = qc_wrfda(rec.sensor, rec.channels, rec.bt, hx, batch.inv, cols, settings,
                                    rec.qc, rec.error)
        batch.qc, batch.error = qc, err
        counts.update(qc=qcounts, locations=rec.n)
        report["sensors"][rec.sensor] = counts
        batches.append(batch)
    return RadianceTerm(batches, shape), report
