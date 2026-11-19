"""Clear-sky infrared brightness temperatures and Jacobians from JCSDA CRTM (opt-in).

    from woof.rtm.crtm import ClearSkyOperator
    with ClearSkyOperator(["abi_g16"]) as crtm:
        result = crtm.simulate("abi_g16", profiles, geometry, when, channels=[8, 9, 10, 14])

The library is the one ``python -m woof.rtm.crtm_build`` writes (woof's shim
linked with a CC0 CRTM v2.4+ tree), found from ``library=`` or
``WOOF_CRTM_LIB`` (the file or its directory).  The coefficients are CRTM's
own ``<sensor>.SpcCoeff.bin``/``.TauCoeff.bin`` and the infrared emissivity
files, found from ``coefficients=`` or ``WOOF_CRTM_COEFFS``.  woof downloads
and ships neither.

CRTM keeps its coefficients in module state, so one operator is open per
process at a time; open it with every sensor the run needs.  Geometry is a
mapping of ``(n,)`` arrays in degrees: ``sensor_zenith``, ``sensor_scan``,
``sensor_azimuth``, ``solar_zenith``, ``solar_azimuth``, ``latitude``,
``longitude``; columns with a sensor zenith above :data:`MAX_SENSOR_ZENITH_DEG`
are refused, since CRTM's transmittance fits stop there.
"""
from __future__ import annotations

import ctypes
import json
import os
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from woof.rtm.profiles import Profiles

LIBRARY_ENV = "WOOF_CRTM_LIB"
COEFFICIENTS_ENV = "WOOF_CRTM_COEFFS"
MAX_SENSOR_ZENITH_DEG = 65.0
_MAX_CHANNELS = 8461  # the largest CRTM sensor (IASI)
_GEOMETRY = ("sensor_zenith", "sensor_scan", "sensor_azimuth", "solar_zenith", "solar_azimuth",
             "latitude", "longitude")
_LOCK = threading.Lock()
_OPEN = None


class CrtmError(RuntimeError):
    """CRTM is not set up, or it refused a call."""


@dataclass
class Simulation:
    """One sensor's clear-sky result for ``n`` columns and the chosen channels."""

    sensor: str
    channels: np.ndarray                 # (nchan,) instrument channel numbers
    brightness_temperature: np.ndarray   # (n, nchan) K
    #: With ``jacobian=True``: dBT/dT (K/K), dBT/dq (K per g/kg) and dBT/dO3
    #: (K/ppmv), ``(n, nchan, nlay)`` top first; dBT/dTskin ``(n, nchan)``.
    k_temperature: np.ndarray | None = None
    k_water_vapour: np.ndarray | None = None
    k_ozone: np.ndarray | None = None
    k_skin_temperature: np.ndarray | None = None


def _library_path(library) -> Path:
    value = library or os.environ.get(LIBRARY_ENV)
    if not value:
        raise CrtmError(f"no CRTM library: build one with `python -m woof.rtm.crtm_build` and set {LIBRARY_ENV}")
    path = Path(value).expanduser()
    if path.is_dir():
        candidates = sorted(path.glob("libwoof_crtm.*"))
        if not candidates:
            raise CrtmError(f"{path}: no libwoof_crtm library (python -m woof.rtm.crtm_build --out {path})")
        path = candidates[0]
    if not path.is_file():
        raise CrtmError(f"{path}: no such CRTM library")
    return path


def _coefficient_path(coefficients) -> Path:
    value = coefficients or os.environ.get(COEFFICIENTS_ENV)
    if not value:
        raise CrtmError(f"no CRTM coefficient directory: set {COEFFICIENTS_ENV} or pass coefficients=")
    path = Path(value).expanduser()
    if not path.is_dir():
        raise CrtmError(f"{path}: not a directory of CRTM coefficients")
    return path


def _pointer(array):
    return array.ctypes.data_as(ctypes.c_void_p)


class ClearSkyOperator:
    """An open CRTM with a fixed set of sensors; use as a context manager or ``close()``."""

    def __init__(self, sensors, *, library=None, coefficients=None):
        global _OPEN
        sensors = [str(s) for s in sensors]
        if not sensors or len(set(sensors)) != len(sensors):
            raise CrtmError(f"name each CRTM sensor once: {sensors}")
        coeff = _coefficient_path(coefficients)
        missing = [f"{s}.{kind}.bin" for s in sensors for kind in ("SpcCoeff", "TauCoeff")
                   if not (coeff / f"{s}.{kind}.bin").is_file()]
        if missing:
            raise CrtmError(f"{coeff}: missing CRTM coefficients {missing}")
        self.library_path = _library_path(library)
        receipt = self.library_path.with_name("build-receipt.json")
        self.receipt = json.loads(receipt.read_text()) if receipt.is_file() else None
        with _LOCK:
            if _OPEN is not None:
                raise CrtmError("a CRTM operator is already open in this process (CRTM keeps module state); "
                                "close it first, or open one operator with every sensor")
            lib = ctypes.CDLL(str(self.library_path))
            for name in ("woof_crtm_init", "woof_crtm_channels", "woof_crtm_destroy", "woof_crtm_clear"):
                getattr(lib, name).restype = ctypes.c_int
            count = ctypes.c_int()
            status = lib.woof_crtm_init(",".join(sensors).encode(), str(coeff).encode(), ctypes.byref(count))
            if status != 0 or count.value != len(sensors):
                raise CrtmError(f"CRTM_Init({sensors}) failed with status {status}")
            _OPEN = self
        self._lib = lib
        self.sensors = tuple(sensors)
        self.coefficients = coeff
        self._channels = {}
        for slot, sensor in enumerate(sensors, 1):
            n = ctypes.c_int()
            channels = np.zeros(_MAX_CHANNELS, np.int32)
            wmo = np.zeros(2, np.int32)
            if lib.woof_crtm_channels(slot, ctypes.byref(n), _pointer(channels), _pointer(wmo)) != 0:
                self.close()
                raise CrtmError(f"CRTM has no channel list for {sensor}")
            self._channels[sensor] = channels[:n.value].copy()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self) -> None:
        global _OPEN
        with _LOCK:
            if _OPEN is self:
                self._lib.woof_crtm_destroy()
                _OPEN = None

    def channels(self, sensor: str) -> np.ndarray:
        """The instrument channel numbers CRTM's coefficients carry for ``sensor``."""

        if sensor not in self._channels:
            raise CrtmError(f"{sensor} is not open (open: {list(self.sensors)})")
        return self._channels[sensor].copy()

    def simulate(self, sensor: str, profiles: Profiles, geometry, when: datetime, *, channels=None,
                 jacobian: bool = False) -> Simulation:
        """Clear-sky brightness temperatures (and Jacobians) for every column."""

        if _OPEN is not self:
            raise CrtmError("this CRTM operator is closed")
        available = self.channels(sensor)
        if channels is None:
            pick = np.arange(available.size)
        else:
            wanted = np.asarray(list(channels), dtype=int)
            absent = sorted(set(wanted.tolist()) - set(available.tolist()))
            if absent:
                raise CrtmError(f"{sensor} coefficients carry channels {available.tolist()}, not {absent}")
            pick = np.array([int(np.flatnonzero(available == c)[0]) for c in wanted])
        n, nlay = profiles.count, profiles.layers
        geo = np.empty((n, 8))
        for k, name in enumerate(_GEOMETRY):
            geo[:, k] = np.broadcast_to(np.asarray(geometry[name], dtype=float), (n,))
        geo[:, 6] = np.mod(geo[:, 6], 360.0)
        geo[:, 7] = profiles.surface_altitude_m
        if not np.all(np.isfinite(geo)):
            raise CrtmError("geometry has non-finite values")
        if np.any(geo[:, 0] > MAX_SENSOR_ZENITH_DEG):
            raise CrtmError(f"sensor zenith above {MAX_SENSOR_ZENITH_DEG} deg; screen those columns first")
        nchan = available.size
        bt = np.zeros((n, nchan))
        shape = (n, nchan, nlay) if jacobian else (1,)
        kt, kq, ko = (np.zeros(shape) for _ in range(3))
        ks = np.zeros((n, nchan) if jacobian else (1,))
        ymd = np.array([when.year, when.month, when.day], np.int32)
        arrays = [profiles.level_pressure_hpa, profiles.pressure_hpa, profiles.temperature_k,
                  profiles.water_vapour_gkg, profiles.ozone_ppmv, np.ascontiguousarray(profiles.surface_array()),
                  np.ascontiguousarray(geo)]
        with _LOCK:
            if _OPEN is not self:
                raise CrtmError("this CRTM operator is closed")
            status = self._lib.woof_crtm_clear(
                ctypes.c_int(self.sensors.index(sensor) + 1), ctypes.c_int(n), ctypes.c_int(nlay),
                *[_pointer(a) for a in arrays], _pointer(ymd), ctypes.c_int(1 if jacobian else 0),
                *[_pointer(a) for a in (bt, kt, kq, ko, ks)])
        if status != 0:
            raise CrtmError(f"CRTM {'K-matrix' if jacobian else 'forward'} call for {sensor} failed with status {status}")
        result = Simulation(sensor=sensor, channels=available[pick], brightness_temperature=bt[:, pick])
        if jacobian:
            result.k_temperature = kt[:, pick]
            result.k_water_vapour = kq[:, pick]
            result.k_ozone = ko[:, pick]
            result.k_skin_temperature = ks[:, pick]
        return result
