"""One in-memory schema for geostationary imager scenes, and their view geometry.

Every imager reader (SatWeave's mosaic, :mod:`woof.obs.satweave`; native
GOES ABI, Himawari AHI and Meteosat readers) returns an :class:`ImagerScene`,
so the radiance operator and the LETKF adapter never know which one produced
it.  Bands are keyed by SatWeave's nominal wavelength in nm (6200, 7330, ...)
and :data:`BANDS` maps each to the channel every instrument carries for it
(docs/dev/WRFDA-3DVAR-PLAN.md section 11.3).

The view geometry of a geostationary imager is closed-form from the
sub-satellite longitude: :func:`geostationary_view` gives the view zenith,
view azimuth and instrument scan (nadir) angle of any ground point on the
WGS84 ellipsoid.  Native readers know their satellite; SatWeave's mosaic
does not publish it, and :mod:`woof.obs.satweave` reconstructs it.

Nothing on a forecast path imports this module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Mapping

import numpy as np

SCENE_SCHEMA = "woof.obs.imager-scene/v1"

#: WGS84 and the geostationary orbit radius (km), as CGMS LRIT/HRIT and the
#: GOES-R product user guide state them.
EARTH_EQUATORIAL_KM = 6378.137
EARTH_POLAR_KM = 6356.7523142
GEOSTATIONARY_RADIUS_KM = 42164.16

#: Flag bits in :attr:`ImagerScene.flags`.
FLAG_OVERLAP = 1          # a second satellite sees the pixel under OVERLAP_ZENITH_DEG
FLAG_LIMB = 2             # view zenith above LIMB_ZENITH_DEG
FLAG_NO_SATELLITE = 4     # no satellite sees the pixel, or none carries what was observed
FLAG_OUT_OF_RANGE = 8     # at least one band value outside its valid range (dropped to NaN)
FLAG_NIGHT = 16           # solar zenith above 90 degrees
FLAG_GEOMETRY_GUESSED = 32  # satellite chosen by reconstruction, not read from the file

OVERLAP_ZENITH_DEG = 70.0
LIMB_ZENITH_DEG = 65.0
#: The horizon test for "a satellite sees this pixel".
HORIZON_ZENITH_DEG = 85.0

BRIGHTNESS_TEMPERATURE = "brightness_temperature"
REFLECTANCE = "reflectance"


@dataclass(frozen=True)
class Band:
    """One SatWeave band and the instrument channel each sensor family uses for it."""

    wavelength_nm: int
    quantity: str
    units: str
    #: Channel numbers per sensor family ("abi", "ahi", "seviri", "fci"); absent = not carried.
    channels: Mapping[str, int]

    @property
    def name(self) -> str:
        return f"mosaic_{self.wavelength_nm}_nm"


def _band(nm, quantity, abi=None, ahi=None, seviri=None, fci=None, *, himawari_only=False):
    channels = {k: v for k, v in (("abi", abi), ("ahi", ahi), ("seviri", seviri), ("fci", fci))
                if v is not None}
    return Band(nm, quantity, "K" if quantity == BRIGHTNESS_TEMPERATURE else "%", channels)


#: The 17 SatWeave bands.  Infrared channel numbers follow each band's
#: ``satellite_sources`` attribute (ABI B08 for 6200 nm, SEVIRI WV6.2, ...);
#: SEVIRI's numbering is its own 1-12 (4 = IR3.9 ... 11 = IR13.4); FCI's is its
#: wavelength order (1 = VIS0.4 ... 16 = IR13.3).
BANDS: Mapping[int, Band] = {b.wavelength_nm: b for b in (
    _band(470, REFLECTANCE, abi=1, ahi=1, fci=1),
    _band(509, REFLECTANCE, ahi=2, fci=2),
    _band(640, REFLECTANCE, abi=2, ahi=3, seviri=1, fci=3),
    _band(863, REFLECTANCE, abi=3, ahi=4, seviri=2, fci=4),
    _band(1370, REFLECTANCE, abi=4, fci=6),
    _band(1610, REFLECTANCE, abi=5, ahi=5, seviri=3, fci=7),
    _band(2250, REFLECTANCE, abi=6, ahi=6, fci=8),
    _band(3830, BRIGHTNESS_TEMPERATURE, abi=7, ahi=7, seviri=4, fci=9),
    _band(6200, BRIGHTNESS_TEMPERATURE, abi=8, ahi=8, seviri=5, fci=10),
    _band(6900, BRIGHTNESS_TEMPERATURE, abi=9, ahi=9),
    _band(7330, BRIGHTNESS_TEMPERATURE, abi=10, ahi=10, seviri=6, fci=11),
    _band(8590, BRIGHTNESS_TEMPERATURE, abi=11, ahi=11, seviri=7, fci=12),
    _band(9620, BRIGHTNESS_TEMPERATURE, abi=12, ahi=12, seviri=8, fci=13),
    _band(10400, BRIGHTNESS_TEMPERATURE, abi=13, ahi=13, fci=14),
    _band(11200, BRIGHTNESS_TEMPERATURE, abi=14, ahi=14, seviri=9),
    _band(12400, BRIGHTNESS_TEMPERATURE, abi=15, ahi=15, seviri=10, fci=15),
    _band(13300, BRIGHTNESS_TEMPERATURE, abi=16, ahi=16, seviri=11, fci=16),
)}


def band(key) -> Band:
    """A band by wavelength (6200) or SatWeave name (``mosaic_6200_nm``)."""

    if isinstance(key, str):
        text = key.removeprefix("mosaic_").removesuffix("_nm")
        try:
            key = int(text)
        except ValueError:
            raise KeyError(f"unknown imager band {key!r}; known: {sorted(BANDS)}") from None
    try:
        return BANDS[int(key)]
    except KeyError:
        raise KeyError(f"unknown imager band {key!r}; known: {sorted(BANDS)}") from None


@dataclass(frozen=True)
class Platform:
    """One geostationary satellite as a scene sees it."""

    name: str              # "GOES-16", "Himawari-9", "Meteosat-11", ...
    sector: str            # "goes_east", "goes_west", "himawari", "meteosat_0", "meteosat_iodc"
    family: str            # "abi", "ahi", "seviri", "fci" (keys of Band.channels)
    crtm_sensor: str       # the CRTM sensor id whose coefficients simulate it
    sub_longitude_deg: float
    #: Why ``crtm_sensor`` is not this satellite's own (empty when it is).
    coefficient_note: str = ""

    def to_payload(self) -> dict:
        return dict(name=self.name, sector=self.sector, family=self.family,
                    crtm_sensor=self.crtm_sensor, sub_longitude_deg=self.sub_longitude_deg,
                    coefficient_note=self.coefficient_note)


def _ecef_ground(lat_deg, lon_deg):
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    lon = np.radians(np.asarray(lon_deg, dtype=float))
    e2 = 1.0 - (EARTH_POLAR_KM / EARTH_EQUATORIAL_KM) ** 2
    n = EARTH_EQUATORIAL_KM / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)
    x = n * np.cos(lat) * np.cos(lon)
    y = n * np.cos(lat) * np.sin(lon)
    z = n * (1.0 - e2) * np.sin(lat)
    return np.stack([x, y, z]), lat, lon


def geostationary_view(lat_deg, lon_deg, sub_longitude_deg: float):
    """View zenith, view azimuth (clockwise from north) and scan angle, in degrees.

    The satellite sits on the equator at ``sub_longitude_deg``, at the
    geostationary radius.  The ground point is on the WGS84 ellipsoid and
    "up" is its geodetic normal.  Pixels the satellite cannot see (behind
    the limb) get a view zenith above 90 degrees.
    """

    ground, lat, lon = _ecef_ground(lat_deg, lon_deg)
    lam = np.radians(sub_longitude_deg)
    sat = GEOSTATIONARY_RADIUS_KM * np.array([np.cos(lam), np.sin(lam), 0.0])
    sat = sat.reshape((3,) + (1,) * (ground.ndim - 1))
    los = sat - ground
    dist = np.sqrt(np.sum(los * los, axis=0))
    up = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    east = np.stack([-np.sin(lon), np.cos(lon), np.zeros_like(lon)])
    north = np.stack([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)])
    cos_zenith = np.sum(los * up, axis=0) / dist
    zenith = np.degrees(np.arccos(np.clip(cos_zenith, -1.0, 1.0)))
    azimuth = np.mod(np.degrees(np.arctan2(np.sum(los * east, axis=0),
                                           np.sum(los * north, axis=0))), 360.0)
    nadir = -sat
    cos_scan = np.sum(nadir * (ground - sat), axis=0) / (GEOSTATIONARY_RADIUS_KM * dist)
    scan = np.degrees(np.arccos(np.clip(cos_scan, -1.0, 1.0)))
    return zenith, azimuth, scan


def solar_angles(when: datetime, lat_deg, lon_deg):
    """Solar zenith and azimuth (degrees) at one time over an array of points."""

    from woof.energy.products import solar_position

    lat = np.asarray(lat_deg, dtype=float)
    lon = np.asarray(lon_deg, dtype=float)
    when = when.astimezone(timezone.utc).replace(tzinfo=None)
    cosz, azimuth = solar_position([when], lat.ravel(), ((lon.ravel() + 180.0) % 360.0) - 180.0)
    zenith = np.degrees(np.arccos(cosz[0])).reshape(lat.shape)
    return zenith, azimuth[0].reshape(lat.shape)


@dataclass
class ImagerScene:
    """One imager's (or one mosaic's) brightness temperatures and reflectances at one time.

    ``values[nm]`` is ``(ny, nx)`` float32 in the band's units (K or %), NaN
    where there is no usable value.  ``platform_index`` indexes
    ``platforms``; -1 where no satellite is attributed.  Angles are degrees.
    ``scan_offset_s`` is each pixel's scan time minus ``time`` where the
    reader knows it, else None (SatWeave publishes no scan time).
    """

    time: datetime
    lat: np.ndarray
    lon: np.ndarray
    values: dict
    platforms: tuple
    platform_index: np.ndarray
    view_zenith: np.ndarray
    view_azimuth: np.ndarray
    scan_angle: np.ndarray
    solar_zenith: np.ndarray
    solar_azimuth: np.ndarray
    flags: np.ndarray
    provenance: dict = field(default_factory=dict)
    scan_offset_s: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.time.tzinfo is None:
            raise ValueError("ImagerScene.time must carry a UTC offset")
        shape = np.shape(self.lat)
        for name in ("lon", "platform_index", "view_zenith", "view_azimuth", "scan_angle",
                     "solar_zenith", "solar_azimuth", "flags"):
            if np.shape(getattr(self, name)) != shape:
                raise ValueError(f"ImagerScene.{name} has shape {np.shape(getattr(self, name))}, lat {shape}")
        for nm, value in self.values.items():
            band(nm)
            if np.shape(value) != shape:
                raise ValueError(f"ImagerScene band {nm} has shape {np.shape(value)}, lat {shape}")

    @property
    def shape(self) -> tuple:
        return tuple(np.shape(self.lat))

    def crtm_sensor(self) -> np.ndarray:
        """The CRTM sensor id per pixel ('' where none is attributed)."""

        names = np.array([p.crtm_sensor for p in self.platforms] + [""], dtype=object)
        return names[np.where(self.platform_index >= 0, self.platform_index, len(self.platforms))]

    def summary(self) -> dict:
        out = dict(schema=SCENE_SCHEMA, time=self.time.isoformat(), shape=list(self.shape),
                   platforms=[p.to_payload() for p in self.platforms],
                   pixels_by_platform={p.name: int(np.sum(self.platform_index == i))
                                       for i, p in enumerate(self.platforms)},
                   unattributed=int(np.sum(self.platform_index < 0)),
                   overlap=int(np.sum(self.flags & FLAG_OVERLAP != 0)),
                   limb=int(np.sum(self.flags & FLAG_LIMB != 0)),
                   bands={})
        for nm in sorted(self.values):
            v = np.asarray(self.values[nm], dtype=float)
            finite = np.isfinite(v)
            out["bands"][str(nm)] = dict(
                quantity=band(nm).quantity, units=band(nm).units, finite=int(finite.sum()),
                mean=float(v[finite].mean()) if finite.any() else None,
                min=float(v[finite].min()) if finite.any() else None,
                max=float(v[finite].max()) if finite.any() else None)
        return out


def utc(value) -> datetime:
    """A timezone-aware UTC datetime from a datetime or an ISO string."""

    if isinstance(value, str):
        text = value.strip().replace("Z", "+00:00")
        value = datetime.fromisoformat(text)
    if not isinstance(value, datetime):
        raise TypeError(f"{value!r} is not a time")
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


SCENE_FILE = "scene.nc"
_SCENE_FIELDS = ("lat", "lon", "platform_index", "view_zenith", "view_azimuth", "scan_angle",
                 "solar_zenith", "solar_azimuth", "flags")


def write_scene_netcdf(directory, scene: ImagerScene, *, licence: str, attribution: str) -> None:
    """Cache ``scene`` as ``directory/scene.nc`` plus ``provenance.json`` (with the file's SHA-256)."""

    import hashlib
    import json
    from pathlib import Path

    from netCDF4 import Dataset

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    import os
    import tempfile

    fd, tmp_name = tempfile.mkstemp(dir=directory, prefix=SCENE_FILE + ".", suffix=".partial")
    os.close(fd)
    tmp = Path(tmp_name)
    ny, nx = scene.shape
    with Dataset(tmp, "w") as ds:
        ds.createDimension("y", ny)
        ds.createDimension("x", nx)
        for name in _SCENE_FIELDS + (("scan_offset_s",) if scene.scan_offset_s is not None else ()):
            value = np.asarray(getattr(scene, name))
            kind = "i2" if value.dtype.kind in "iu" else ("f8" if name in ("lat", "lon") else "f4")
            ds.createVariable(name, kind, ("y", "x"), zlib=True)[:] = value
        for nm, value in scene.values.items():
            var = ds.createVariable(band(nm).name, "f4", ("y", "x"), zlib=True, fill_value=np.float32(np.nan))
            var.units = band(nm).units
            var[:] = value
        ds.schema = SCENE_SCHEMA
        ds.time = scene.time.isoformat()
        ds.platforms = json.dumps([p.to_payload() for p in scene.platforms])
        ds.licence = licence
        ds.attribution = attribution
    os.replace(tmp, directory / SCENE_FILE)
    provenance = dict(scene.provenance, cache_sha256=hashlib.sha256(
        (directory / SCENE_FILE).read_bytes()).hexdigest())
    fd, ptmp = tempfile.mkstemp(dir=directory, prefix="provenance.json.", suffix=".partial")
    with os.fdopen(fd, "w") as fh:
        fh.write(json.dumps(provenance, indent=2, default=str) + "\n")
    os.replace(ptmp, directory / "provenance.json")
    scene.provenance = provenance


def read_scene_netcdf(directory) -> ImagerScene:
    """A scene :func:`write_scene_netcdf` cached; refused when the file's digest has changed."""

    import hashlib
    import json
    from pathlib import Path

    from netCDF4 import Dataset

    directory = Path(directory)
    provenance = json.loads((directory / "provenance.json").read_text())
    digest = hashlib.sha256((directory / SCENE_FILE).read_bytes()).hexdigest()
    if digest != provenance.get("cache_sha256"):
        raise ValueError(f"{directory / SCENE_FILE}: digest {digest} does not match its provenance "
                         f"({provenance.get('cache_sha256')}); delete the entry or refresh it")
    by_name = {band(nm).name: nm for nm in BANDS}
    with Dataset(directory / SCENE_FILE) as ds:
        ds.set_auto_mask(False)
        fields = {name: np.asarray(ds[name][:]) for name in _SCENE_FIELDS}
        offset = np.asarray(ds["scan_offset_s"][:]) if "scan_offset_s" in ds.variables else None
        values = {by_name[k]: np.asarray(v[:], dtype="f4") for k, v in ds.variables.items() if k in by_name}
        platforms = tuple(Platform(**p) for p in json.loads(ds.platforms))
        when = utc(ds.time)
    return ImagerScene(time=when, values=values, platforms=platforms, provenance=provenance,
                       scan_offset_s=offset, **fields)
